use anyhow::{Context as _, Result, ensure};
use gpui::WgslPostprocessPass;
use metal::{Device, MTLPixelFormat, RenderPipelineState};
use std::{
    collections::VecDeque,
    sync::{Arc, Weak},
};

const VERTEX: &str = r#"
#include <metal_stdlib>
vertex metal::float4 scoped_effect_vertex(uint index [[vertex_id]]) {
    float x = float((index << 1u) & 2u);
    float y = float(index & 2u);
    return metal::float4(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}
"#;

struct CachedProgram {
    source: Weak<str>,
    entry: Arc<str>,
    uniform_size: usize,
    pipeline: RenderPipelineState,
}

pub(super) struct Compiler {
    device: Device,
    vertex: Option<metal::Function>,
    programs: VecDeque<CachedProgram>,
}

impl Compiler {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            vertex: None,
            programs: VecDeque::new(),
        }
    }

    pub fn program(
        &mut self,
        pass: &WgslPostprocessPass,
        uniform_size: usize,
    ) -> Result<RenderPipelineState> {
        // 只在后台访问缓存；弱源键使关闭效果后的源程序能在下次准备时退出缓存。
        self.programs
            .retain(|program| program.source.strong_count() != 0);
        if let Some(program) = self.programs.iter().find(|program| {
            program
                .source
                .upgrade()
                .is_some_and(|source| Arc::ptr_eq(&source, &pass.source))
                && program.entry == pass.entry
                && program.uniform_size == uniform_size
        }) {
            return Ok(program.pipeline.clone());
        }
        let (source, entry) = translate(pass, uniform_size)?;
        let options = metal::CompileOptions::new();
        options.set_language_version(metal::MTLLanguageVersion::V2_1);
        options.set_fast_math_enabled(false);
        if self.vertex.is_none() {
            let library = self
                .device
                .new_library_with_source(VERTEX, &options)
                .map_err(|error| anyhow::anyhow!("compile effect vertex: {error}"))?;
            self.vertex = Some(
                library
                    .get_function("scoped_effect_vertex", None)
                    .map_err(|error| anyhow::anyhow!("load effect vertex: {error}"))?,
            );
        }
        let library = self
            .device
            .new_library_with_source(&source, &options)
            .map_err(|error| {
                anyhow::anyhow!(
                    "compile effect MSL: {}",
                    error.chars().take(4096).collect::<String>()
                )
            })?;
        let fragment = library
            .get_function(&entry, None)
            .map_err(|error| anyhow::anyhow!("load effect fragment: {error}"))?;
        let descriptor = metal::RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(self.vertex.as_deref());
        descriptor.set_fragment_function(Some(&fragment));
        descriptor.set_sample_count(1);
        descriptor
            .color_attachments()
            .object_at(0)
            .context("effect color attachment missing")?
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        let pipeline = self
            .device
            .new_render_pipeline_state(&descriptor)
            .map_err(|error| anyhow::anyhow!("prepare effect pipeline: {error}"))?;
        if self.programs.len() == 64 {
            self.programs.pop_front();
        }
        self.programs.push_back(CachedProgram {
            source: Arc::downgrade(&pass.source),
            entry: pass.entry.clone(),
            uniform_size,
            pipeline: pipeline.clone(),
        });
        Ok(pipeline)
    }
}

pub(super) fn translate(
    pass: &WgslPostprocessPass,
    uniform_size: usize,
) -> Result<(String, String)> {
    let (module, info) = gpui::validate_postprocess_wgsl(&pass.source, &pass.entry, uniform_size)?;
    let mut resources = naga::back::msl::EntryPointResources::default();
    resources.resources.insert(
        naga::ResourceBinding {
            group: 0,
            binding: 0,
        },
        naga::back::msl::BindTarget {
            buffer: Some(0),
            ..Default::default()
        },
    );
    resources.resources.insert(
        naga::ResourceBinding {
            group: 0,
            binding: 1,
        },
        naga::back::msl::BindTarget {
            texture: Some(0),
            ..Default::default()
        },
    );
    let mut options = naga::back::msl::Options {
        lang_version: (2, 1),
        fake_missing_bindings: false,
        ..Default::default()
    };
    options
        .per_entry_point_map
        .insert(pass.entry.to_string(), resources);
    let pipeline = naga::back::msl::PipelineOptions {
        entry_point: Some((naga::ShaderStage::Fragment, pass.entry.to_string())),
        ..Default::default()
    };
    let (source, translated) = naga::back::msl::write_string(&module, &info, &options, &pipeline)?;
    ensure!(
        source.len() <= 1024 * 1024,
        "effect MSL exceeds its preparation limit"
    );
    // 入口经过 Naga 的保留字和重名处理，原始 WGSL 名称不一定是可链接的 MSL 名称。
    let entry = translated
        .entry_point_names
        .into_iter()
        .next()
        .context("effect MSL entry missing")??;
    Ok((source, entry))
}
