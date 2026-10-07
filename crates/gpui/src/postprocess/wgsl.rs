//! WGSL 后处理的共享 ABI 校验；原生后端只负责目标语言编译。
use anyhow::{Result, ensure};

/// Validates the shared frame/surface ABI before a native backend compiles it.
/// Call from preparation workers, not the UI/paint path.
pub fn validate_postprocess_wgsl(
    source: &str,
    entry: &str,
    uniform_size: usize,
) -> Result<(naga::Module, naga::valid::ModuleInfo)> {
    super::validate_source(source, entry)?;
    super::validate_uniform_size(uniform_size)?;
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| anyhow::anyhow!(error.emit_to_string(source)))?;
    let mut uniform = false;
    let mut surface = false;
    for (_, global) in module.global_variables.iter() {
        if global.space == naga::AddressSpace::Private && global.binding.is_none() {
            continue;
        }
        let binding = global
            .binding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("effect global requires the declared ABI"))?;
        ensure!(binding.group == 0, "effect binding group changed");
        match binding.binding {
            0 => {
                ensure!(
                    !uniform
                        && global.space == naga::AddressSpace::Uniform
                        && matches!(module.types[global.ty].inner, naga::TypeInner::Struct { span, .. } if span as usize == uniform_size),
                    "effect uniform ABI changed"
                );
                uniform = true;
            },
            1 => {
                ensure!(
                    !surface
                        && global.space == naga::AddressSpace::Handle
                        && matches!(
                            module.types[global.ty].inner,
                            naga::TypeInner::Image {
                                dim: naga::ImageDimension::D2,
                                arrayed: false,
                                class: naga::ImageClass::Sampled {
                                    kind: naga::ScalarKind::Float,
                                    multi: false
                                }
                            }
                        ),
                    "effect surface ABI changed"
                );
                surface = true;
            },
            _ => anyhow::bail!("effect binding is not admitted"),
        }
    }
    ensure!(uniform && surface, "effect frame and surface bindings are required");
    ensure!((1..=8).contains(&module.entry_points.len()), "invalid effect entry count");
    for point in &module.entry_points {
        ensure!(
            point.stage == naga::ShaderStage::Fragment
                && point.function.arguments.len() <= 1
                && point.function.arguments.iter().all(|argument| matches!(
                    argument.binding,
                    Some(naga::Binding::BuiltIn(naga::BuiltIn::Position { .. }))
                )),
            "effect input is fragment position"
        );
        let result = point
            .function
            .result
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("effect output required"))?;
        ensure!(
            matches!(result.binding, Some(naga::Binding::Location { location: 0, .. }))
                && matches!(
                    module.types[result.ty].inner,
                    naga::TypeInner::Vector {
                        size: naga::VectorSize::Quad,
                        scalar: naga::Scalar { kind: naga::ScalarKind::Float, width: 4 }
                    }
                ),
            "effect output must be location0 vec4<f32>"
        );
    }
    ensure!(module.entry_points.iter().any(|point| point.name == entry), "effect entry is missing");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)?;
    Ok((module, info))
}
