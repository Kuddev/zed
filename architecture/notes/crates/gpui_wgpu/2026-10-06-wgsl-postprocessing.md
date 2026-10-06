# WGSL postprocessing on native wgpu

## Status

Work in progress. The native executor has hardware Vulkan pixel and lifetime
evidence. Complete common, WGPU, and Windows renderer source checks pass on
Windows MSVC; integrated window acceptance remains pending. Product dependencies
still pin the preceding validated revision.

## Context

The scoped scene primitive already exists, but its preparation descriptor carries
Direct3D bytecode. Native wgpu needs WGSL modules, explicit surface-copy usages and
queue-ordered frame data. Applying an effect at final presentation would include
unrelated UI; copying the frame to the CPU would add avoidable work.

## Decision

- Add a separate WGSL preparation descriptor/factory, retaining the precompiled
  Direct3D interface. Both descriptors share surface, uniform and pass limits.
- Prepare two full-resolution textures, one uniform buffer and bounded shared
  programs off the UI thread. Retain resources through the existing GPU completion
  and budget-retirement mechanism.
- Require a copyable unorm surface and retain the existing native Vulkan admission;
  this does not re-enable unqualified GL preparation or implement Metal/WebGPU.
- Submit preceding scene work at the effect barrier, submit the effect, then resume
  subsequent scene work with load semantics. Separate submissions ensure later
  queue uniform writes do not change earlier uses of the same owner in one frame.
- Clear excluded input pixels and restrict both copies to the visible content mask.
  Keep native ownership, cancellation and device-epoch checks at adoption.

## Rejected alternatives

- Reinterpret WGSL as Direct3D bytecode or alter existing callers silently.
- Use a screenshot/readback as the product input to every frame.
- Keep a texture per stage or reduce terminal text resolution.
- Issue all uniform writes before one final submit: repeated owner use would read
  the last uniform value for earlier draws as well.

## Consequences

Effects add queue submissions but not extra per-stage texture storage. Native
compiler/driver allocations remain outside the known texture/uniform byte count.
The application must opt into the new factory only after renderer and product
qualification; existing dependency pins are not changed by this checkpoint.

## Validation

A native Vulkan fixture imports the actual common descriptor and native executor.
It verifies ordered pixels, untouched outside regions, excluded-input transparency,
independent consecutive uniform updates, eight-stage constant texture admission,
insufficient budget, cancellation, obsolete epochs and acknowledged retirement.
The Windows MSVC source check at revision
`067d5225d6e701fc6d6804a207fd738d6f6ac5a0` passed with Rust 1.97.1:
`cargo check --locked -p gpui -p gpui_wgpu -p gpui_windows --lib`.
Evidence: [native source run](https://github.com/Kuddev/zed/actions/runs/37496042949).
The qualification branch uses a hosted runner because inherited CI requires
upstream-only infrastructure. A short Cargo home and Git long-path support avoid
a dependency checkout failure without changing dependencies or the lockfile.

Integrated scene/surface execution, the expanded BGRA/application-ABI fixture,
and other-platform acceptance remain pending. Source compilation is not hardware
or window execution and does not complete backend qualification.

The expanded fixture now lives in `native_postprocess/tests.rs`, using the real
crate types and implementation; the local standalone launcher reuses this module.
It is explicitly ignored in ordinary test runs because execution requires a
hardware Vulkan adapter and the product's actual UTF-8 ABI file through
`PEBREL_EFFECT_ABI`. Its ignored status is not a passing hardware result.

At `f62c414139da6f47ae18023e5cf85e316d1339c3`,
[run 37501147011](https://github.com/Kuddev/zed/actions/runs/37501147011) passed
complete source checking and `cargo test --locked -p gpui_wgpu --lib --no-run`.
The retained executable and manifest record the source revision and SHA256.
This removes the need to compile the fixture on the physical acceptance machine;
the RGBA/BGRA, eight-pass, clipping, ABI, cancellation, and retirement assertions
still require explicit execution there.

## Supersedes

None. Extends the existing scoped primitive without changing its Direct3D contract.

## Revisit when

Integrated frame measurements justify a different submission strategy, or another
native backend needs a verified preparation and lifetime contract.
