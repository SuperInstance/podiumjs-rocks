# PLATO Vector Compute Extension

Extension to podiumjs-rocks (original by Grig/vdmo).

## What was added
- `src/compute/wgsl_similarity.wgsl` — WGSL compute kernel for PLATO tile embedding cosine similarity
- `src/compute/ComputePipeline.ts` — WebGPU compute pipeline using Grig's WebGPUContext pattern

## Usage
```typescript
import { WebGPUContext } from '../core/WebGPUContext';
import { ComputePipeline } from './compute/ComputePipeline';

const ctx = new WebGPUContext(canvas);
await ctx.init();
const pipeline = new ComputePipeline(ctx);
const results = await pipeline.search(queryEmbedding, allEmbeddings, 10);
```

## Architecture
Follows Grig's module structure:
- `core/` — WebGPU context, uniforms, post-processing (Grig)
- `compute/` — PLATO similarity compute pipeline (this extension)
- `shaders/` — Shader management (Grig)
- `geometry/` — 3D geometry (Grig)

## Integration
The compute pipeline feeds results to the rendering pipeline:
1. Compute kernel finds K nearest tile neighbors
2. Results are passed to the render pipeline
3. Grig's ShaderManager renders tiles as 3D points on the knowledge manifold
