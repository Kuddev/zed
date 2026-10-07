//! 后台 WGSL → 原生程序转换；绘制与 GPU 资源生命周期继续使用既有后处理实现。
use anyhow::{Context as _, Result, ensure};
use gpui::{
    BackgroundShaderCancellation, PostprocessDescriptor, WgslPostprocessDescriptor,
    WgslPostprocessPass,
};
use std::{
    ffi::CString,
    sync::{Arc, Mutex, OnceLock, Weak},
};

const MAX_CACHED_PROGRAMS: usize = 64;
const MAX_BYTECODE_BYTES: usize = 64 * 1024;

struct CachedProgram {
    source: Weak<str>,
    entry: Arc<str>,
    uniform_size: usize,
    bytecode: Arc<[u8]>,
}
static PROGRAMS: OnceLock<Mutex<Vec<CachedProgram>>> = OnceLock::new();

pub(super) fn compile(
    descriptor: &WgslPostprocessDescriptor,
    cancellation: &BackgroundShaderCancellation,
) -> Result<Option<PostprocessDescriptor>> {
    descriptor.validate()?;
    let mut programs = PROGRAMS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .map_err(|_| anyhow::anyhow!("effect compiler cache poisoned"))?;
    // 弱引用不延长源文件生命周期；下次编译回收过期项。保留字节码上限为 4 MiB。
    programs.retain(|program| program.source.strong_count() != 0);
    let mut passes = Vec::with_capacity(descriptor.passes.len());
    for pass in descriptor.passes.iter() {
        if cancellation.is_cancelled() {
            return Ok(None);
        }
        let source = Arc::downgrade(&pass.source);
        if let Some(program) = programs.iter().find(|program| {
            program.source.ptr_eq(&source)
                && program.entry == pass.entry
                && program.uniform_size == descriptor.uniform_size
        }) {
            passes.push(program.bytecode.clone());
            continue;
        }
        let bytecode = compile_pass(pass, descriptor.uniform_size)?;
        if cancellation.is_cancelled() {
            return Ok(None);
        }
        // 缓存满时只淘汰缓存引用，现有描述符仍持有程序，不把缓存容量变成用户功能限制。
        if programs.len() == MAX_CACHED_PROGRAMS {
            programs.remove(0);
        }
        programs.push(CachedProgram {
            source,
            entry: pass.entry.clone(),
            uniform_size: descriptor.uniform_size,
            bytecode: bytecode.clone(),
        });
        passes.push(bytecode);
    }
    Ok(Some(PostprocessDescriptor {
        size: descriptor.size,
        uniform_size: descriptor.uniform_size,
        directx_passes: passes.into(),
    }))
}

fn compile_pass(pass: &WgslPostprocessPass, uniform_size: usize) -> Result<Arc<[u8]>> {
    let (module, info) = gpui::validate_postprocess_wgsl(&pass.source, &pass.entry, uniform_size)?;
    let mut options = naga::back::hlsl::Options {
        shader_model: naga::back::hlsl::ShaderModel::V5_0,
        fake_missing_bindings: false,
        ..Default::default()
    };
    for binding in [0, 1] {
        options.binding_map.insert(
            naga::ResourceBinding { group: 0, binding },
            naga::back::hlsl::BindTarget { register: 0, ..Default::default() },
        );
    }
    let pipeline = naga::back::hlsl::PipelineOptions {
        entry_point: Some((naga::ShaderStage::Fragment, pass.entry.to_string())),
    };
    let mut hlsl = String::new();
    let reflection = naga::back::hlsl::Writer::new(&mut hlsl, &options, &pipeline)
        .write(&module, &info, None)?;
    ensure!(hlsl.len() <= 256 * 1024, "translated effect exceeds its byte budget");
    let entry = reflection
        .entry_point_names
        .into_iter()
        .next()
        .context("missing native effect entry")?
        .map_err(|error| anyhow::anyhow!("{error:?}"))?;
    compile_native(&hlsl, &entry)
}

fn compile_native(hlsl: &str, entry: &str) -> Result<Arc<[u8]>> {
    use windows::{
        Win32::Graphics::Direct3D::{Fxc::*, ID3DInclude},
        core::{PCSTR, s},
    };
    let entry = CString::new(entry)?;
    let mut bytecode = None;
    let mut diagnostics = None;
    let result = unsafe {
        D3DCompile(
            hlsl.as_ptr().cast(),
            hlsl.len(),
            PCSTR::null(),
            None,
            None::<&ID3DInclude>,
            PCSTR(entry.as_ptr().cast()),
            s!("ps_4_1"),
            D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut bytecode,
            Some(&mut diagnostics),
        )
    };
    if let Err(error) = result {
        let detail = diagnostics
            .map(|blob| unsafe {
                String::from_utf8_lossy(std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast(),
                    blob.GetBufferSize().min(4096),
                ))
                .into_owned()
            })
            .unwrap_or_default();
        anyhow::bail!("effect native compilation failed: {error}: {detail}");
    }
    let blob = bytecode.context("native compiler returned no bytecode")?;
    let bytes = unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    };
    ensure!(
        bytes.len() <= MAX_BYTECODE_BYTES && bytes.starts_with(b"DXBC"),
        "invalid effect bytecode"
    );
    Ok(Arc::from(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &str = r#"
struct Params { factor: vec4<f32> }
@group(0) @binding(0) var<uniform> frame: Params;
@group(0) @binding(1) var surface: texture_2d<f32>;
@fragment fn first(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(surface, vec2<i32>(p.xy), 0) * frame.factor;
}
@fragment fn second() -> @location(0) vec4<f32> { return frame.factor; }
"#;
    fn descriptor() -> WgslPostprocessDescriptor {
        let source: Arc<str> = SOURCE.into();
        WgslPostprocessDescriptor {
            size: gpui::size(gpui::DevicePixels(4), gpui::DevicePixels(2)),
            uniform_size: 16,
            passes: ["first", "second"]
                .into_iter()
                .map(|entry| WgslPostprocessPass { source: source.clone(), entry: entry.into() })
                .collect::<Vec<_>>()
                .into(),
        }
    }
    #[test]
    fn native_entries_preserve_order_and_reuse_live_sources() {
        let descriptor = descriptor();
        let first = compile(&descriptor, &Default::default()).unwrap().unwrap();
        first.validate().unwrap();
        assert_eq!(first.directx_passes.len(), 2);
        assert_ne!(first.directx_passes[0], first.directx_passes[1]);
        let mut resized = descriptor.clone();
        resized.size.width = gpui::DevicePixels(8);
        let second = compile(&resized, &Default::default()).unwrap().unwrap();
        for (first, second) in first.directx_passes.iter().zip(second.directx_passes.iter()) {
            assert!(Arc::ptr_eq(first, second));
        }
    }
    #[test]
    fn cancellation_and_abi_errors_do_not_publish_a_partial_chain() {
        let cancellation = BackgroundShaderCancellation::default();
        cancellation.cancel();
        assert!(compile(&descriptor(), &cancellation).unwrap().is_none());
        let mut invalid = descriptor();
        invalid.uniform_size = 32;
        assert!(compile(&invalid, &Default::default()).is_err());
        let mut passes = invalid.passes.to_vec();
        passes[1].entry = "missing".into();
        invalid.uniform_size = 16;
        invalid.passes = passes.into();
        assert!(compile(&invalid, &Default::default()).is_err());
    }
}
