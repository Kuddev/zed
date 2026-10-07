# WGSL preparation on the Windows backend

## Status

Implemented; source compilation and focused native compiler checks passed.
The product dependency pins and application caller have not changed in this step.

## Context

The WGPU backend accepts ordered WGSL sources, while the Windows backend accepted
only precompiled fragment bytecode. Requiring the application to select a native
compiler duplicates backend knowledge and prevents one portable product entry.
Replacing the existing bytecode interface would unnecessarily break current callers.

## Evidence

The application already validates the same frame/surface binding and fragment
entry contract used by the WGPU executor. It keeps compiled programs across
surface resize. A source-only native entry must preserve that reuse rather than
compiling each time an owner changes extent.

## Decision

- Keep the existing bytecode factory. Add the existing WGSL factory contract to
  DirectX, sharing capture, cancellation, device epoch, adoption, and retirement.
- Compile only on the preparation worker, before native texture allocation.
- Move the WGPU ABI validator to the descriptor-owning common crate behind the
  optional `wgsl-postprocess` feature. Both backends use that one implementation.
  Unrelated common-crate users do not enable the optional parser automatically.
- Keep the existing HLSL register mapping, shader model, compiler flags, diagnostics
  limit, and 64 KiB bytecode limit when producing Windows native programs.
- Reuse compiled bytecode for the same live source allocation, entry, and uniform
  size. Weak source keys do not keep source text alive. Dead entries are pruned on
  the next compilation; a 64-entry FIFO cap bounds retained bytecode to 4 MiB.
  Eviction drops only the cache reference, not programs used by active descriptors.
  Weak references to unsized source allocations can retain allocation storage
  until pruning: account for up to another 8 MiB of source storage at the transport
  limit, plus small entry metadata. This is a bound, not a measured typical cost.

## Rejected alternatives

- Removing the bytecode interface before migrating its callers.
- Copying the ABI validator into another backend.
- Making DirectX depend on the entire WGPU backend to obtain a parser.
- Performing parsing or compilation inside paint, or recompiling on every resize.
- An unbounded permanent shader cache or a generic compiler plugin framework.

## Consequences

The new entry is compatible with the portable source transport without changing
native rendering behavior. Naga 29.0.4 was already present in the dependency graph;
the common optional feature and Windows backend now consume the same version.
Compiler caching is separate from the existing per-device GPU-program cache and
does not alter texture or compiler-job admission budgets.

## Validation

At `d34e803beb6bfdf77451f052fae84dca9e606755`,
[run 37567979636](https://github.com/Kuddev/zed/actions/runs/37567979636) passed
the complete common/WGPU/Windows source check and two native compiler tests.
Those tests executed actual compilation, ordered entries, live-source reuse
across resize, cancellation, and ABI/entry rejection. Product caller migration
and its full native validation are separate evidence.

## Supersedes

None. Extends the precompiled Windows interface and shares the previously
qualified WGPU input contract.

## Revisit when

Another backend consumes the same WGSL entry, or measured preparation costs
justify changing the bounded cache policy without changing ownership semantics.
