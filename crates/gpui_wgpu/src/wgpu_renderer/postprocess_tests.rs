//! 原生表面初始化与真实场景批次；离屏读回不冒充可见窗口呈现验收。
use super::*;
use gpui::{
    BackgroundExecutor, ContentMask, PaintPostprocess, PlatformAtlas, PostprocessFeedback, Quad,
    StreamImageBudget, StreamImageBudgets, StreamImageHandle, TestDispatcher,
    WgslPostprocessDescriptor, WgslPostprocessPass, point, rgb, size,
};
use raw_window_handle::{
    DisplayHandle, HandleError, RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowHandle,
    WindowsDisplayHandle,
};
use std::{num::NonZeroIsize, time::Duration};
use windows::{
    Win32::{
        Foundation::{HINSTANCE, HWND},
        System::LibraryLoader::GetModuleHandleW,
        UI::WindowsAndMessaging::{
            CreateWindowExW, DestroyWindow, IsWindowVisible, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW,
            WS_POPUP,
        },
    },
    core::w,
};

struct NativeWindow(HWND);
impl NativeWindow {
    fn new() -> Self {
        // 不设 WS_VISIBLE，也不调用 ShowWindow/SetForegroundWindow；原生表面仍有真实 HWND。
        let hwnd = unsafe {
            CreateWindowExW(
                WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
                w!("STATIC"),
                w!("native scene acceptance"),
                WS_POPUP,
                0,
                0,
                12,
                4,
                None,
                None,
                None,
                None,
            )
        }
        .unwrap();
        assert!(!unsafe { IsWindowVisible(hwnd) }.as_bool());
        Self(hwnd)
    }
}
impl Drop for NativeWindow {
    fn drop(&mut self) {
        if let Err(error) = unsafe { DestroyWindow(self.0) } {
            eprintln!("native acceptance window cleanup failed: {error}");
        }
    }
}

#[derive(Clone, Debug)]
struct NativeHandle {
    hwnd: isize,
    instance: isize,
}
impl HasWindowHandle for NativeHandle {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let mut handle = Win32WindowHandle::new(NonZeroIsize::new(self.hwnd).unwrap());
        handle.hinstance = NonZeroIsize::new(self.instance);
        // NativeWindow 的所有者在 renderer/context 之后销毁，借用不延长 HWND 的生命周期。
        Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
    }
}
impl HasDisplayHandle for NativeHandle {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Ok(unsafe {
            DisplayHandle::borrow_raw(RawDisplayHandle::Windows(WindowsDisplayHandle::new()))
        })
    }
}

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
"#;

fn region(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds::new(
        point(ScaledPixels(x), ScaledPixels(y)),
        size(ScaledPixels(width), ScaledPixels(height)),
    )
}
fn quad(bounds: Bounds<ScaledPixels>, color: u32) -> Quad {
    Quad {
        bounds,
        content_mask: ContentMask { bounds },
        background: rgb(color).into(),
        ..Default::default()
    }
}
fn effect(
    owner: &StreamImageHandle,
    bounds: Bounds<ScaledPixels>,
    mask: Bounds<ScaledPixels>,
    factor: f32,
) -> PaintPostprocess {
    let mut uniforms = vec![0; 16];
    uniforms[..4].copy_from_slice(&factor.to_le_bytes());
    PaintPostprocess {
        order: 0,
        bounds,
        content_mask: ContentMask { bounds: mask },
        owner: owner.clone(),
        uniforms: uniforms.into(),
        feedback: PostprocessFeedback::default(),
    }
}

fn render_and_read(renderer: &mut WgpuRenderer, scene: &Scene) -> Vec<[u8; 4]> {
    renderer.atlas.before_frame();
    renderer.ensure_intermediate_textures();
    let resources = renderer.resources();
    let texture = resources.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("actual renderer scene readback target"),
        size: wgpu::Extent3d { width: 12, height: 4, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: renderer.surface_config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let globals = GlobalParams { viewport_size: [12., 4.], premultiplied_alpha: 0, pad: 0 };
    resources.queue.write_buffer(&resources.globals_buffer, 0, bytemuck::bytes_of(&globals));
    let view = texture.create_view(&Default::default());
    // 直接调用生产 record_frame，覆盖其真实批次、前置提交、效果调用和 Load 续画。
    renderer.record_frame(scene, &texture, &view).unwrap();
    let resources = renderer.resources();
    let output = resources.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scene oracle readback"),
        size: 256 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = resources.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        texture.size(),
    );
    resources.queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    output.slice(..).map_async(wgpu::MapMode::Read, move |result| sender.send(result).unwrap());
    resources
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        })
        .unwrap();
    receiver.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    let data = output.slice(..).get_mapped_range();
    let mut pixels = Vec::new();
    for y in 0..4 {
        for x in 0..12 {
            let offset = y * 256 + x * 4;
            let mut pixel: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
            if texture.format() == wgpu::TextureFormat::Bgra8Unorm {
                pixel.swap(0, 2);
            }
            pixels.push(pixel);
        }
    }
    drop(data);
    output.unmap();
    for effect in &scene.postprocesses {
        assert!(effect.feedback.take_error().is_none());
    }
    assert!(renderer.last_error.lock().unwrap().is_none());
    pixels
}

#[test]
#[ignore = "requires hardware Vulkan; constructs a hidden non-activating native window"]
fn native_surface_scene_order_and_replay() {
    let window = NativeWindow::new();
    let instance: HINSTANCE = unsafe { GetModuleHandleW(None) }.unwrap().into();
    let handle = NativeHandle { hwnd: window.0.0 as isize, instance: instance.0 as isize };
    let context = Rc::new(RefCell::new(None));
    let mut renderer = WgpuRenderer::new(
        context.clone(),
        &handle,
        WgpuSurfaceConfig {
            size: size(DevicePixels(12), DevicePixels(4)),
            transparent: false,
            preferred_present_mode: None,
        },
        None,
    )
    .unwrap();
    let info = renderer.adapter_info.clone();
    println!(
        "scene hardware adapter: {info:?}; surface format: {:?}",
        renderer.surface_config.format
    );
    assert_eq!(info.backend, wgpu::Backend::Vulkan);
    assert_ne!(info.device_type, wgpu::DeviceType::Cpu);
    assert!(
        renderer
            .surface_config
            .usage
            .contains(wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST)
    );
    let budget = StreamImageBudget::new(4 * 2 * 8 + 16);
    let owner = StreamImageHandle::from_platform_atlas(
        renderer.atlas.clone(),
        BackgroundExecutor::new(Arc::new(TestDispatcher::new(0))),
        StreamImageBudgets::new(StreamImageBudget::new(4 * 2 * 8 + 16), budget.clone()),
    );
    let preparation = owner
        .prepare_postprocess_wgsl(
            WgslPostprocessDescriptor {
                size: size(DevicePixels(4), DevicePixels(2)),
                uniform_size: 16,
                passes: ["invert", "scale"]
                    .into_iter()
                    .map(|entry| WgslPostprocessPass { source: SOURCE.into(), entry: entry.into() })
                    .collect::<Vec<_>>()
                    .into(),
            },
            Default::default(),
        )
        .unwrap();
    let prepared = std::thread::spawn(move || preparation.run()).join().unwrap().unwrap().unwrap();
    owner.adopt_postprocess(prepared).unwrap();
    let full = region(0., 0., 12., 4.);
    let mut scene = Scene::default();
    scene.push_layer(full);
    scene.insert_primitive(quad(full, 0x0000ff));
    scene.insert_primitive(effect(&owner, region(1., 1., 4., 2.), region(2., 1., 2., 2.), 0.5));
    scene.pop_layer();
    scene.insert_primitive(quad(region(2., 1., 1., 1.), 0xff0000));
    scene.finish();
    let expected = (0..48)
        .map(|index| {
            let (x, y) = (index % 12, index / 12);
            if x == 2 && y == 1 {
                [255, 0, 0, 255]
            } else if (2..4).contains(&x) && (1..3).contains(&y) {
                [128, 255, 0, 255]
            } else {
                [0, 0, 255, 255]
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(render_and_read(&mut renderer, &scene), expected);
    let mut replay = Scene::default();
    replay.replay(0..scene.len(), &scene);
    replay.finish();
    assert_eq!(render_and_read(&mut renderer, &replay), expected);
    let mut repeated = Scene::default();
    repeated.insert_primitive(quad(full, 0x0000ff));
    repeated.insert_primitive(effect(&owner, region(0., 0., 4., 2.), full, 0.5));
    repeated.insert_primitive(effect(&owner, region(6., 0., 4., 2.), full, 0.25));
    repeated.finish();
    let pixels = render_and_read(&mut renderer, &repeated);
    assert_eq!(pixels[0], [128, 255, 0, 255]);
    assert_eq!(pixels[6], [64, 255, 0, 255]);
    // 显式完成原生退役后再释放模拟 executor 的句柄，不在模拟 UI 线程等待 GPU。
    let receipt = renderer.atlas.retire_stream_image(owner.id()).unwrap().unwrap();
    std::thread::spawn(move || receipt.wait()).join().unwrap().unwrap();
    assert_eq!(budget.used(), 0);
    println!(
        "native surface setup, actual scene dispatch, clipping, layer exit, replay and repeated-owner uniforms passed; visible presentation not exercised"
    );
}
