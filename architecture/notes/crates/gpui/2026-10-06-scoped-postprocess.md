# Scoped post-processing and native resource ownership

## Status

Native integration in progress. The common window/scene API and Direct3D pass
executor exist; application settings, terminal snapshots and other native backend
implementations are not yet enabled by this change.

## Context

A background shader executes before the terminal's GPU text batches. It therefore
cannot sample the terminal output. Applying a filter only at window presentation
would instead include menus, other panes and unrelated application UI.

## Evidence

Scenes batch primitives by spatial draw order and may inherit an enclosing layer's
order. A new effect needs a barrier that survives leaving a nested layer and scene
replay. Native texture submission and Rust owner destruction also remain distinct
events; extra pass resources must survive the input owner's last GPU completion.

## Decision

- Insert a scoped post-processing primitive after its input content. Assign an
  ordering barrier and offset subsequent logical orders, including enclosing-layer
  draws. Retain the effect owner and immutable frame data through scene replay.
- Prepare native objects on a worker; adopt only on the matching UI thread/device
  epoch with an uncancelled receipt. Unsupported adapters return explicit errors.
- Use two exact-size BGRA targets for one through eight native passes. Alternate
  targets and copy the final result back to the visible region; all passes use the
  same local pixel coordinates. Do not lower text resolution or allocate per pass.
- Apply the content mask to the input copy as well as the output copy. Clear the
  remaining input pixels so a shader cannot sample excluded underlying content.
- Reuse stream-image preparation admission, native program sharing and fence
  retirement. Attach auxiliary textures, programs, uniforms and their byte lease
  to that retirement owner instead of inventing a second completion/quarantine path.
- Preserve renderer bindings around the effect; report failures to its owning view
  and retain ordinary window rendering. Backpressure skips the effect for a frame
  without allocating additional GPU slots.
- The application source interface remains WGSL. Native Direct3D bytecode is a
  backend representation, not a user-facing source-language compatibility layer.

## Rejected alternatives

- Read back a screenshot to the CPU every animation frame.
- Sample before the native terminal batches have executed.
- Apply every effect to the whole window after unrelated UI has drawn.
- Keep one intermediate texture per pass, or release auxiliary resources before
  the input owner's fence is acknowledged.
- Clip only the output: a native pixel regression demonstrated that excluded input
  pixels were still readable until input copying used the same visible region.

## Consequences

Known texture bytes are twice the physical surface size plus the declared constant
buffer. Admission remains explicit; shader instructions and driver allocations are
not bounded by that byte count. The owning application must prepare replacement
resources on geometry changes and present errors using its localization contract.

## Validation

Current GPUI and Windows library sources pass Rust metadata compilation. A native
hardware-D3D fixture compiles WGSL through Naga and verifies two-pass pixel results,
eight-pass constant storage, clipping, excluded-input sampling, outside-surface
preservation, invalid uniforms, render-target restoration and fence-owned budget
release. A failing excluded-input oracle preceded the corrected implementation.
Scene ordering/replay tests are added but still await the full GPUI test build.
These checks do not establish real terminal interaction or other-platform execution.

## Supersedes

None. Background-only rendering remains separate and retains its existing behavior.

## Revisit when

The application adapter, another native backend or real frame-time/resource evidence
changes the ordering, geometry, admission or completion assumptions.
