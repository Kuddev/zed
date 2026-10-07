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

Visible swapchain presentation and other-platform acceptance remain pending.
Source compilation alone does not complete backend qualification.

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
are exercised by explicitly selecting that test there.

On 2026-10-07 the retained executable was run on an NVIDIA GeForce RTX 5060 Laptop
GPU, driver 581.57, Vulkan backend. The explicit hardware test passed for both
RGBA8 and BGRA8, including the actual 4304-byte product ABI, eight-pass pixels,
excluded input, negative-origin clipping, uniform sequencing, cancellation,
obsolete epochs, wrong-owner adoption, and acknowledged resource retirement.
The executable SHA256 was
`7b649cd2a0be7edde6b706d04a764af58cc756211467545c075d9915e0296cf4`;
the product ABI SHA256 was
`86a48243b1cbc035baafac4b43b6aaac5fa212c1da11d0371a9bc37f07bf3e28`.
One test containing those scenarios passed; its 2.11-second test duration is not
an application animation benchmark. This exercised the production native
executor, not `WgpuRenderer::draw`, scene dispatch, or a product window.

The separate scene test at `4e1448b84844b16376ccd72bc02df4a84987dcde`
then passed on the same physical Vulkan GPU with a BGRA8 surface. It creates a
hidden, non-activating native window, negotiates the actual surface capabilities,
uses the public owner factory/adoption interface, and invokes production
`WgpuRenderer::record_frame` with offscreen pixel readback. Assertions cover
effect/overlay ordering across layer exit, clipping, scene replay, repeated-owner
uniform isolation, and native retirement. It does not present a visible swapchain
frame or execute the product's terminal view.

The first scene attempt rejected red=127 where the oracle demanded 128. The
[Vulkan conversion rules](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/chapters/fundamentals.adoc)
allow either adjacent integer for fractional UNORM conversion. Only that
quantized channel now accepts 127/128 or 63/64; alpha, endpoints, overlays, and
outside-region pixels remain exact. A deterministic positive/negative oracle
test passed remotely before the hardware rerun. Production rendering code was
not changed to satisfy the oracle. The successful scene executable SHA256 is
`e6715e901964118837bb40044f65dd3fc04228f0171dd2971b96fae0a2845778`.

## Supersedes

None. Extends the existing scoped primitive without changing its Direct3D contract.

## Revisit when

Integrated frame measurements justify a different submission strategy, or another
native backend needs a verified preparation and lifetime contract.
