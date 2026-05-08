# PodiumJS Rocks + PLATO Vector Compute Extension

**Original:** PodiumJS Rocks by [Grig (vdmo)](https://github.com/vdmo/podiumjs-rocks) — WebGPU-based 3D rendering for interactive planes.

**Extended by:** SuperInstance fleet — adding GPU-accelerated vector similarity search for PLATO knowledge tiles.

## What This Is

A WebGPU rendering framework (original) + PLATO vector similarity compute pipeline (extension).

### Original (Grig)
- `src/core/WebGPUContext.ts` — GPU adapter/device management
- `src/core/Podium.ts` — Main rendering class
- `src/core/PostProcessor.ts` — Post-processing effects
- `src/core/UniformManager.ts` — Uniform buffer management
- `src/shaders/ShaderManager.ts` — Shader compilation and caching
- `src/geometry/` — 3D geometry helpers
- `src/textures/` — Texture management

### Extension (SuperInstance)
- `src/compute/wgsl_similarity.wgsl` — WGSL compute kernel for cosine similarity search across PLATO tile embeddings
- `src/compute/ComputePipeline.ts` — WebGPU compute pipeline using Grig's WebGPUContext pattern

## Usage

```typescript
import { WebGPUContext } from '../core/WebGPUContext';
import { ComputePipeline } from './compute/ComputePipeline';

const ctx = new WebGPUContext(canvas);
await ctx.init();

const pipeline = new ComputePipeline(ctx);
// Search 10K embeddings for the 10 nearest to a query
const results = await pipeline.search(queryEmbedding, allEmbeddings, 10);
```

## Why This Matters

Every PLATO tile gets a 384-dimensional embedding. When a new tile is submitted, its embedding is compared against all existing embeddings to find semantically similar tiles. This is a GPU-perfect workload: N threads, each computing one cosine similarity, all in parallel.

The WebGPU compute kernel dispatches 1 thread per embedding. For 10K embeddings at 384 dimensions: ~3.8M FMAs, <10μs on any modern GPU.
