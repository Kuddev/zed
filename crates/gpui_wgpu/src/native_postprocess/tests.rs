//! 显式硬件验收：普通 CI 只编译，实际运行需 Vulkan GPU 和产品原始 ABI 文件。
use super::{NativePostprocess, PostprocessDevice};
use gpui::{
    BackgroundShaderCancellation, Bounds, DevicePixels, ScaledPixels, StreamImageBudget,
    StreamImageBudgets, WgslPostprocessDescriptor, WgslPostprocessPass, point, size,
};
use std::sync::{Arc, atomic::AtomicBool};

const SOURCE: &str = r#"
struct Params { factor: vec4<f32> }
@group(0) @binding(0) var<uniform> frame: Params;
@group(0) @binding(1) var surface: texture_2d<f32>;
@fragment fn invert(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(surface, vec2<i32>(p.xy), 0);
    return vec4<f32>(vec3<f32>(c.a) - c.rgb, c.a);
}
@fragment fn scale(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(surface, vec2<i32>(p.xy), 0);
    return vec4<f32>(c.r * frame.factor.x, c.g, c.b, c.a);
}
@fragment fn excluded() -> @location(0) vec4<f32> {
    return textureLoad(surface, vec2<i32>(0, 0), 0);
}
"#;
fn descriptor(entries: &[&str]) -> WgslPostprocessDescriptor {
    WgslPostprocessDescriptor {
        size: size(DevicePixels(4), DevicePixels(2)),
        uniform_size: 16,
        passes: entries
            .iter()
            .map(|entry| WgslPostprocessPass { source: SOURCE.into(), entry: (*entry).into() })
            .collect::<Vec<_>>()
            .into(),
    }
}
fn rectangle(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
    Bounds::new(point(ScaledPixels(x), ScaledPixels(y)), size(ScaledPixels(w), ScaledPixels(h)))
}
fn uniform(factor: f32) -> [u8; 16] {
    let mut bytes = [0; 16];
    bytes[..4].copy_from_slice(&factor.to_le_bytes());
    bytes
}
fn screen(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("postprocess oracle screen"),
        size: wgpu::Extent3d { width: 12, height: 4, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // 两种纹理格式写入相同的逻辑颜色，读回时再统一为 RGBA 进行像素比较。
    let pixel = if format == wgpu::TextureFormat::Bgra8Unorm {
        [100, 60, 40, 255]
    } else {
        [40, 60, 100, 255]
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixel.repeat(48),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(48), rows_per_image: None },
        texture.size(),
    );
    texture
}
fn read(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<[u8; 4]> {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oracle readback only"),
        size: 256 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let data = buffer.slice(..).get_mapped_range();
    let mut result = Vec::new();
    for y in 0..4 {
        for x in 0..12 {
            let i = y * 256 + x * 4;
            let mut pixel: [u8; 4] = data[i..i + 4].try_into().unwrap();
            if texture.format() == wgpu::TextureFormat::Bgra8Unorm {
                pixel.swap(0, 2);
            }
            result.push(pixel);
        }
    }
    drop(data);
    buffer.unmap();
    result
}
fn prepare(
    device: &PostprocessDevice,
    owner: u64,
    desc: WgslPostprocessDescriptor,
    budgets: StreamImageBudgets,
    format: wgpu::TextureFormat,
) -> NativePostprocess {
    let factory = device.factory(owner, desc, format, budgets, Default::default()).unwrap();
    let prepared = std::thread::spawn(move || factory.run()).join().unwrap().unwrap().unwrap();
    device.adopt(owner, prepared).unwrap()
}
fn retire(image: NativePostprocess) {
    if let Some(completion) = image.retire() {
        std::thread::spawn(move || completion.wait()).join().unwrap().unwrap();
    }
}
#[test]
#[ignore = "requires a hardware Vulkan adapter and PEBREL_EFFECT_ABI pointing to the product ABI"]
fn native_vulkan_postprocess_contract() {
    let abi_path = std::env::var_os("PEBREL_EFFECT_ABI").expect("provide the product ABI file");
    let application_abi = std::fs::read_to_string(&abi_path).expect("read UTF-8 product ABI");
    println!("product ABI file: {}", std::path::Path::new(&abi_path).display());
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .unwrap();
    println!("hardware adapter: {:?}", adapter.get_info());
    assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let (device, queue) = (Arc::new(device), Arc::new(queue));
    let native =
        PostprocessDevice::new(device.clone(), queue.clone(), Arc::new(AtomicBool::new(false)))
            .unwrap();
    for format in [wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Bgra8Unorm] {
        println!("validating texture format: {format:?}");
        let bytes = 4 * 2 * 8 + 16;
        let budget = StreamImageBudget::new(bytes);
        let budgets = StreamImageBudgets::new(StreamImageBudget::new(bytes), budget.clone());
        let mut image =
            prepare(&native, 1, descriptor(&["invert", "scale"]), budgets.clone(), format);
        assert_eq!(budget.used(), bytes);
        let texture = screen(&device, &queue, format);
        assert!(
            image
                .render(&texture, rectangle(1., 1., 4., 2.), rectangle(2., 1., 2., 2.), &[0; 32])
                .is_err()
        );
        image
            .render(&texture, rectangle(1., 1., 4., 2.), rectangle(2., 1., 2., 2.), &uniform(0.5))
            .unwrap();
        let pixels = read(&device, &queue, &texture);
        for y in 0..4 {
            for x in 0..12 {
                let pixel = pixels[y * 12 + x];
                if (2..4).contains(&x) && (1..3).contains(&y) {
                    assert!((107..=108).contains(&pixel[0]));
                    assert_eq!(&pixel[1..], &[195, 155, 255]);
                } else {
                    assert_eq!(pixel, [40, 60, 100, 255]);
                }
            }
        }
        retire(image);
        assert_eq!(budget.used(), 0);
        let texture = screen(&device, &queue, format);
        let mut image = prepare(&native, 2, descriptor(&["excluded"]), budgets.clone(), format);
        image
            .render(&texture, rectangle(1., 1., 4., 2.), rectangle(2., 1., 2., 2.), &uniform(1.))
            .unwrap();
        assert_eq!(read(&device, &queue, &texture)[14], [0, 0, 0, 0]);
        retire(image);
        assert_eq!(budget.used(), 0);
        let texture = screen(&device, &queue, format);
        let mut image = prepare(&native, 3, descriptor(&["scale"]), budgets.clone(), format);
        image
            .render(&texture, rectangle(0., 0., 4., 2.), rectangle(0., 0., 12., 4.), &uniform(0.5))
            .unwrap();
        image
            .render(&texture, rectangle(6., 0., 4., 2.), rectangle(0., 0., 12., 4.), &uniform(0.25))
            .unwrap();
        let pixels = read(&device, &queue, &texture);
        assert_eq!(pixels[0], [20, 60, 100, 255]);
        assert_eq!(pixels[6], [10, 60, 100, 255]);
        retire(image);
        assert_eq!(budget.used(), 0);
        let mut image = prepare(&native, 4, descriptor(&["invert"; 8]), budgets.clone(), format);
        assert_eq!(budget.used(), bytes);
        let texture = screen(&device, &queue, format);
        image
            .render(&texture, rectangle(0., 0., 4., 2.), rectangle(0., 0., 12., 4.), &uniform(1.))
            .unwrap();
        assert!(read(&device, &queue, &texture).iter().all(|pixel| *pixel == [40, 60, 100, 255]));
        retire(image);
        assert_eq!(budget.used(), 0);
        let low = StreamImageBudgets::new(
            StreamImageBudget::new(bytes - 1),
            StreamImageBudget::new(bytes - 1),
        );
        let factory =
            native.factory(5, descriptor(&["invert"]), format, low, Default::default()).unwrap();
        assert!(std::thread::spawn(move || factory.run()).join().unwrap().is_err());
        let cancelled = BackgroundShaderCancellation::default();
        let factory = native
            .factory(6, descriptor(&["invert"]), format, budgets.clone(), cancelled.clone())
            .unwrap();
        cancelled.cancel();
        assert!(std::thread::spawn(move || factory.run()).join().unwrap().unwrap().is_none());
        assert_eq!(budget.used(), 0);
        let factory = native
            .factory(7, descriptor(&["invert"]), format, budgets.clone(), Default::default())
            .unwrap();
        let prepared = std::thread::spawn(move || factory.run()).join().unwrap().unwrap().unwrap();
        native.invalidate().unwrap();
        assert!(native.adopt(7, prepared).is_err());
        assert_eq!(budget.used(), 0);
        let cancellation = BackgroundShaderCancellation::default();
        let factory = native
            .factory(8, descriptor(&["invert"]), format, budgets.clone(), cancellation.clone())
            .unwrap();
        let other = cancellation.clone();
        assert!(
            std::thread::spawn(move || factory.run_with_allocation_gate(move || other.cancel()))
                .join()
                .unwrap()
                .unwrap()
                .is_none()
        );
        assert_eq!(budget.used(), 0);
        let factory = native
            .factory(9, descriptor(&["invert"]), format, budgets.clone(), Default::default())
            .unwrap();
        let prepared = std::thread::spawn(move || factory.run()).join().unwrap().unwrap().unwrap();
        assert!(native.adopt(10, prepared).is_err());
        assert_eq!(budget.used(), 0);
        let mut abi = descriptor(&["app_frame"]);
        abi.uniform_size = (13 + 256) * 16;
        abi.passes = vec![WgslPostprocessPass {
            source: format!(
                "{}\n@fragment fn app_frame()->@location(0) vec4<f32>{{return frame.palette[7u];}}",
                application_abi
            )
            .into(),
            entry: "app_frame".into(),
        }]
        .into();
        let large_bytes = abi.texture_bytes().unwrap() + abi.uniform_size as u64;
        let large_budget = StreamImageBudget::new(large_bytes);
        let mut image = prepare(
            &native,
            11,
            abi,
            StreamImageBudgets::new(StreamImageBudget::new(large_bytes), large_budget.clone()),
            format,
        );
        let mut payload = vec![0; 4304];
        for (i, value) in [0.2f32, 0.4, 0.6, 1.0].iter().enumerate() {
            let offset = (13 + 7) * 16 + i * 4;
            payload[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let texture = screen(&device, &queue, format);
        image
            .render(&texture, rectangle(0., 0., 4., 2.), rectangle(0., 0., 12., 4.), &payload)
            .unwrap();
        assert_eq!(read(&device, &queue, &texture)[0], [51, 102, 153, 255]);
        retire(image);
        assert_eq!(large_budget.used(), 0);
        // 验证窗口左边缘裁剪使用局部坐标，而不是把负原点转换成无符号偏移。
        let texture = screen(&device, &queue, format);
        let mut image = prepare(&native, 12, descriptor(&["invert"]), budgets.clone(), format);
        image
            .render(&texture, rectangle(-1., 1., 4., 2.), rectangle(0., 0., 12., 4.), &uniform(1.))
            .unwrap();
        let pixels = read(&device, &queue, &texture);
        for y in 0..4 {
            for x in 0..12 {
                let expected = if x < 3 && (1..3).contains(&y) {
                    [215, 195, 155, 255]
                } else {
                    [40, 60, 100, 255]
                };
                assert_eq!(pixels[y * 12 + x], expected);
            }
        }
        retire(image);
        assert_eq!(budget.used(), 0);
        println!(
            "format {format:?}: native pixels, application ABI, uniform sequencing, masking, budgets, cancellation, epoch and retirement passed"
        );
    }
}
