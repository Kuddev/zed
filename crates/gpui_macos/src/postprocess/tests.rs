use super::*;
use crate::metal_renderer::{InstanceBufferPool, MetalRenderer};
use gpui::{
    BackgroundExecutor, ContentMask, DevicePixels, PlatformAtlas, PostprocessFeedback, Quad, Scene,
    StreamImageBudget, StreamImageHandle, TestDispatcher, WgslPostprocessPass, rgb,
};
use std::{
    io::Write,
    process::{Command, Stdio},
};

const ABI: &str = r#"
struct Params { factor: vec4<f32> }
@group(0) @binding(0) var<uniform> frame: Params;
@group(0) @binding(1) var surface: texture_2d<f32>;
"#;
const PASSES: &str = r#"
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
    let source: Arc<str> = format!("{ABI}\n{PASSES}").into();
    WgslPostprocessDescriptor {
        size: size(DevicePixels(4), DevicePixels(2)),
        uniform_size: 16,
        passes: entries
            .iter()
            .map(|entry| WgslPostprocessPass {
                source: source.clone(),
                entry: (*entry).into(),
            })
            .collect::<Vec<_>>()
            .into(),
    }
}

fn compile_msl(pass: &WgslPostprocessPass, uniform_size: usize) {
    let (source, _) = compiler::translate(pass, uniform_size).unwrap();
    let mut child = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-std=macos-metal2.1",
            "-x",
            "metal",
            "-c",
            "-",
            "-o",
            "/dev/null",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn selected_entries_compile_with_the_native_msl_compiler() {
    for pass in descriptor(&["invert", "scale", "excluded"]).passes.iter() {
        compile_msl(pass, 16);
    }
    let missing = WgslPostprocessPass {
        entry: "missing".into(),
        ..descriptor(&["invert"]).passes[0].clone()
    };
    assert!(compiler::translate(&missing, 16).is_err());
    assert!(compiler::translate(&descriptor(&["invert"]).passes[0], 32).is_err());
}

#[test]
#[ignore = "requires the actual product ABI via PEBREL_EFFECT_ABI and the native MSL compiler"]
fn product_abi_compiles_with_the_native_msl_compiler() {
    let abi = std::fs::read_to_string(std::env::var("PEBREL_EFFECT_ABI").unwrap()).unwrap();
    let source: Arc<str> = format!(
        r#"{abi}
@fragment fn palette_and_sample(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{
    return sample_surface(p.xy / frame.viewport.xy) * frame.palette[frame.flags.x % 256u];
}}
"#
    )
    .into();
    compile_msl(
        &WgslPostprocessPass {
            source,
            entry: "palette_and_sample".into(),
        },
        4304,
    );
}

fn rectangle(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds::new(
        point(ScaledPixels(x), ScaledPixels(y)),
        size(ScaledPixels(width), ScaledPixels(height)),
    )
}

fn owner(renderer: &MetalRenderer, budget: &Arc<StreamImageBudget>) -> StreamImageHandle {
    StreamImageHandle::from_platform_atlas(
        renderer.sprite_atlas().clone(),
        BackgroundExecutor::new(Arc::new(TestDispatcher::new(0))),
        StreamImageBudgets::new(budget.clone(), StreamImageBudget::new(1024 * 1024)),
    )
}

fn prepare(owner: &StreamImageHandle, descriptor: WgslPostprocessDescriptor) {
    let work = owner
        .prepare_postprocess_wgsl(descriptor, Default::default())
        .unwrap();
    let prepared = std::thread::spawn(move || work.run())
        .join()
        .unwrap()
        .unwrap()
        .unwrap();
    owner.adopt_postprocess(prepared).unwrap();
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

fn background() -> Scene {
    let full = rectangle(0.0, 0.0, 12.0, 4.0);
    let mut scene = Scene::default();
    scene.insert_primitive(Quad {
        bounds: full,
        content_mask: ContentMask { bounds: full },
        background: rgb(0x0000ff).into(),
        ..Default::default()
    });
    scene
}

fn read(renderer: &mut MetalRenderer, scene: &Scene) -> image::RgbaImage {
    let pixels = renderer
        .render_scene_to_image(scene, size(DevicePixels(12), DevicePixels(4)))
        .unwrap();
    for effect in &scene.postprocesses {
        assert!(effect.feedback.take_error().is_none());
    }
    pixels
}

#[test]
#[ignore = "requires a native Metal device; runs production scene rendering and pixel readback"]
fn native_metal_scene_and_retirement() {
    objc::rc::autoreleasepool(|| {
        assert!(
            metal::Device::system_default().is_some(),
            "native Metal acceptance requires a Metal device"
        );
        let mut renderer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(InstanceBufferPool::default())));
        let atlas = renderer.sprite_atlas().clone();
        let device = atlas.postprocess.device.device.clone();
        println!("Metal device: {}", device.name());
        assert!(atlas.supports_postprocess_wgsl());
        let budget = StreamImageBudget::new(64);
        let image = owner(&renderer, &budget);
        prepare(&image, descriptor(&["invert", "scale"]));
        assert_eq!(budget.used(), 64);
        let full = rectangle(0.0, 0.0, 12.0, 4.0);
        let mut scene = background();
        scene.push_layer(full);
        scene.insert_primitive(gpui::Primitive::Postprocess(effect(
            &image,
            rectangle(1.0, 1.0, 4.0, 2.0),
            rectangle(2.0, 1.0, 2.0, 2.0),
            0.0,
        )));
        scene.pop_layer();
        scene.insert_primitive(Quad {
            bounds: rectangle(2.0, 1.0, 1.0, 1.0),
            content_mask: ContentMask { bounds: full },
            background: rgb(0xff0000).into(),
            ..Default::default()
        });
        scene.finish();
        let pixels = read(&mut renderer, &scene);
        for y in 0..4 {
            for x in 0..12 {
                let expected = if x == 2 && y == 1 {
                    [255, 0, 0, 255]
                } else if (2..4).contains(&x) && (1..3).contains(&y) {
                    [0, 255, 0, 255]
                } else {
                    [0, 0, 255, 255]
                };
                assert_eq!(pixels.get_pixel(x, y).0, expected, "pixel {x},{y}");
            }
        }
        let mut replay = Scene::default();
        replay.replay(0..scene.len(), &scene);
        replay.finish();
        assert_eq!(read(&mut renderer, &replay), pixels);
        let mut repeated = background();
        repeated.insert_primitive(gpui::Primitive::Postprocess(effect(
            &image,
            rectangle(0.0, 0.0, 4.0, 2.0),
            full,
            0.0,
        )));
        repeated.insert_primitive(gpui::Primitive::Postprocess(effect(
            &image,
            rectangle(6.0, 0.0, 4.0, 2.0),
            full,
            1.0,
        )));
        repeated.finish();
        let pixels = read(&mut renderer, &repeated);
        assert_eq!(pixels.get_pixel(0, 0).0, [0, 255, 0, 255]);
        assert_eq!(pixels.get_pixel(6, 0).0, [255, 255, 0, 255]);
        atlas.retire_stream_image(image.id()).unwrap();
        assert_eq!(budget.used(), 0);

        for entries in [vec!["invert"; 8], vec!["excluded"]] {
            let image = owner(&renderer, &budget);
            prepare(&image, descriptor(&entries));
            assert_eq!(budget.used(), 64);
            let mut scene = background();
            scene.insert_primitive(gpui::Primitive::Postprocess(effect(
                &image,
                rectangle(-2.0, 0.0, 4.0, 2.0),
                full,
                1.0,
            )));
            scene.finish();
            let pixels = read(&mut renderer, &scene);
            let expected = if entries.len() == 8 {
                [0, 0, 255, 255]
            } else {
                [0, 0, 0, 0]
            };
            assert_eq!(pixels.get_pixel(0, 0).0, expected);
            assert_eq!(pixels.get_pixel(2, 0).0, [0, 0, 255, 255]);
            atlas.retire_stream_image(image.id()).unwrap();
            assert_eq!(budget.used(), 0);
        }

        let image = owner(&renderer, &budget);
        let cancellation = BackgroundShaderCancellation::default();
        let work = image
            .prepare_postprocess_wgsl(descriptor(&["invert"]), cancellation.clone())
            .unwrap();
        cancellation.cancel();
        assert!(
            std::thread::spawn(move || work.run())
                .join()
                .unwrap()
                .unwrap()
                .is_none()
        );
        let work = image
            .prepare_postprocess_wgsl(descriptor(&["invert"]), Default::default())
            .unwrap();
        let prepared = std::thread::spawn(move || work.run())
            .join()
            .unwrap()
            .unwrap()
            .unwrap();
        image.invalidate_background_preparations_for_test().unwrap();
        assert!(image.adopt_postprocess(prepared).is_err());
        assert_eq!(budget.used(), 0);

        let undersized = owner(&renderer, &StreamImageBudget::new(63));
        let work = undersized
            .prepare_postprocess_wgsl(descriptor(&["invert"]), Default::default())
            .unwrap();
        assert!(
            std::thread::spawn(move || work.run())
                .join()
                .unwrap()
                .is_err()
        );
        let wrong_owner = owner(&renderer, &budget);
        let work = image
            .prepare_postprocess_wgsl(descriptor(&["invert"]), Default::default())
            .unwrap();
        let prepared = std::thread::spawn(move || work.run())
            .join()
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(wrong_owner.adopt_postprocess(prepared).is_err());
        assert_eq!(budget.used(), 0);

        let mut reused = descriptor(&["invert"]);
        prepare(&image, reused.clone());
        let pipeline = atlas.postprocess.images.borrow()[&image.id()]
            .resources
            .pipelines[0]
            .clone();
        atlas.retire_stream_image(image.id()).unwrap();
        reused.size.width = DevicePixels(3);
        prepare(&image, reused);
        assert_eq!(
            pipeline.as_ptr(),
            atlas.postprocess.images.borrow()[&image.id()]
                .resources
                .pipelines[0]
                .as_ptr()
        );
        atlas.retire_stream_image(image.id()).unwrap();

        let abi = std::fs::read_to_string(std::env::var("PEBREL_EFFECT_ABI").unwrap()).unwrap();
        let source: Arc<str> = format!("{abi}\n@fragment fn palette_tail() -> @location(0) vec4<f32> {{ return frame.palette[255]; }}").into();
        prepare(
            &image,
            WgslPostprocessDescriptor {
                size: size(DevicePixels(4), DevicePixels(2)),
                uniform_size: 4304,
                passes: vec![WgslPostprocessPass {
                    source,
                    entry: "palette_tail".into(),
                }]
                .into(),
            },
        );
        let mut frame = effect(&image, rectangle(0.0, 0.0, 4.0, 2.0), full, 0.0);
        let mut uniforms = vec![0; 4304];
        for (index, value) in [0.0f32, 1.0, 0.0, 1.0].into_iter().enumerate() {
            uniforms[4288 + index * 4..4292 + index * 4].copy_from_slice(&value.to_le_bytes());
        }
        frame.uniforms = uniforms.into();
        let mut scene = background();
        scene.insert_primitive(gpui::Primitive::Postprocess(frame));
        scene.finish();
        assert_eq!(
            read(&mut renderer, &scene).get_pixel(0, 0).0,
            [0, 255, 0, 255]
        );
        atlas.retire_stream_image(image.id()).unwrap();
        assert_eq!(budget.used(), 0);

        // 编码后、提交前移除 owner，租约仍由真实 command buffer 的完成回调持有。
        let image = owner(&renderer, &budget);
        prepare(&image, descriptor(&["invert"]));
        let screen = target(&device, &descriptor(&["invert"])).unwrap();
        let queue = device.new_command_queue();
        let command = queue.new_command_buffer();
        let uniform = device.new_buffer(256, metal::MTLResourceOptions::StorageModeShared);
        let frame = effect(&image, rectangle(0.0, 0.0, 4.0, 2.0), full, 1.0);
        clear(command, &screen).unwrap();
        atlas
            .postprocess
            .render(command, &screen, &frame, &uniform, 0)
            .unwrap();
        atlas.retire_stream_image(image.id()).unwrap();
        assert_eq!(budget.used(), 64);
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
        assert_eq!(budget.used(), 0);
        println!(
            "Metal scene ordering, replay, clipping, independent uniforms, eight-pass storage, cancellation, stale adoption and completion-owned retirement passed"
        );
    });
}
