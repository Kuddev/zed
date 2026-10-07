# Scoped WGSL postprocessing on Metal

## Status

Implemented on the renderer feature branch. Native compilation and pixel/lifetime
qualification are pending; product dependency pins have not changed in this step.

## Context

The common scene records a barrier after a terminal pane's content and before
application overlays. Metal previously reported that operation as unsupported.
Its ordinary window drawable is framebuffer-only, and the product frame ABI is
4304 bytes: an inline fragment argument would exceed Metal's small-data limit.

## Evidence

- Apple documents that [framebufferOnly](https://developer.apple.com/documentation/quartzcore/cametallayer/framebufferonly)
  prevents sampling and pixel read/write access. It can retain display optimizations
  for ordinary scenes, while postprocessing needs copyable drawables.
- [setFragmentBytes](https://developer.apple.com/documentation/metal/mtlrendercommandencoder/setfragmentbytes(_:length:index:))
  is for data smaller than 4 KiB. Larger data must use a native buffer binding.
- [addCompletedHandler](https://developer.apple.com/documentation/metal/mtlcommandbuffer/addcompletedhandler(_:))
  runs after GPU command execution and must be registered before commit.
- The existing renderer's instance writer hands out aligned, nonoverlapping slices,
  flushes managed storage, and returns its pooled buffer after command completion.
- The pinned Naga 29.0.4 backend exposes explicit MSL entry-resource mappings and
  translated entry names. metal-rs 0.33 marks the retained native objects Send/Sync.

## Decision

- Reuse the common WGSL validator and preparation admission. On a worker with an
  autorelease pool, translate the selected entry to MSL 2.1, bind frame buffer 0
  and surface texture 0, compile native programs, and allocate two full-resolution
  BGRA textures. No file read, compilation or pipeline allocation occurs in paint.
- Keep an atlas-owned compiler cache, accessed only by workers. It retains at most
  64 native pipelines, keyed by weak source allocation, entry and uniform size.
  Dead sources prune on the next preparation. Live-source reuse survives resize.
  The cache's weak source storage is bounded by 8 MiB plus metadata; native compiler
  and pipeline allocations have no asserted byte bound.
- Capture atlas identity, its owning thread, cancellation and device epoch in every
  preparation receipt. Adoption rejects stale, foreign, cancelled or duplicate
  ownership without replacing an existing effect.
- End the preceding scene encoder, clear excluded input, copy the visible pane
  region, alternate passes between the two targets, and copy back only that region.
  Resume normal scene rendering with load semantics. The product path has no CPU
  pixel readback and does not reduce text resolution.
- Store each effect invocation's uniforms in its own existing instance-buffer
  slice. This supports the full product ABI and repeated owner use in one command
  buffer without later writes changing earlier draws. Effect leases charge the
  two textures; the existing bounded instance pool owns frame-parameter storage.
- Retain the native resources and their common budget lease in the actual command
  buffer's completion handler. Removing an atlas owner releases only its reference.
  Completion releases its retained reference; a GPU error first revokes admission,
  marks the atlas unavailable and reports through the effect feedback. No second
  polling thread or retirement timer is introduced. A command buffer that never
  completes continues retaining its admitted resources.
- Only scenes containing effects turn off framebuffer-only mode for normal windows.
  Existing test-support screenshot behavior remains copyable. Renderer destruction
  closes admission and drops atlas references while in-flight callbacks retain theirs.

## Rejected alternatives

- Inline bytes for the 4304-byte ABI, or overwriting a shared uniform buffer several
  times before submitting the command buffer.
- Permanently disabling framebuffer-only optimization for ordinary product windows.
- Filtering the whole final window, reducing text resolution, CPU frame readback,
  or clipping only the output while leaving excluded input readable.
- Treating present/scheduled notification or logical owner removal as GPU completion.
- Adding a GLSL compatibility frontend or a separate product controller for Metal.

## Consequences

The existing product controller can use Metal after qualification and exact-revision
pin updates. The backend keeps its own native preparation and execution details;
source ordering, activation, animation modes and persistence remain product-owned.
Per-atlas caches and copyable drawables add costs requiring representative product
measurement. This change does not implement Metal background media or video decoding.

## Validation

Native compiler tests translate selected entries, reject ABI/entry mismatches and
compile generated MSL with Xcode. A separate explicit test reads the product ABI
from a hash-verified exact source revision. The native device test exercises actual
scene ordering, overlays, replay, masked and negative-origin input, repeated-owner
uniform isolation, the tail of the 4304-byte ABI, eight-pass storage, insufficient
budget, cancellation, stale/foreign receipts, source reuse and completion-owned
budget release. Device absence is an error in the explicitly requested native test,
not a silent passing result. These tests are pending execution at this checkpoint.
Headless pixels do not establish visible window, foreground latency or endurance
acceptance. The current Windows machine is not a Metal qualification host.

## Supersedes

Extends the [scoped postprocess contract](../gpui/2026-10-06-scoped-postprocess.md)
with a Metal implementation. Existing DirectX and WGPU paths remain in their owners.

## Revisit when

Actual runtime costs justify different command encoding or cache admission, the
product ABI gains resources beyond the shared validator's contract, or a native
streaming adapter needs to share Metal completion ownership.
