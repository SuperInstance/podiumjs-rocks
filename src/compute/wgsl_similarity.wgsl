// PLATO Vector Similarity — WebGPU compute shader
// Forked extension to podiumjs-rocks.
// Original podiumjs-rocks by Grig (vdmo).
// Added: cosine similarity kernel for PLATO tile embedding search.

struct Uniforms {
    num_embeddings: u32,
    dim: u32,
    top_k: u32,
};

struct SimilarityResult {
    index: u32,
    score: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> query_embedding: array<f32>;
@group(0) @binding(2) var<storage, read> embeddings: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<SimilarityResult>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= uniforms.num_embeddings) { return; }
    
    var dot = 0.0f;
    var norm_q = 0.0f;
    var norm_e = 0.0f;
    
    for (var d = 0u; d < uniforms.dim; d = d + 1u) {
        let q = query_embedding[d];
        let e = embeddings[idx * uniforms.dim + d];
        dot = dot + q * e;
        norm_q = norm_q + q * q;
        norm_e = norm_e + e * e;
    }
    
    let similarity = dot / (sqrt(norm_q) * sqrt(norm_e) + 1e-8);
    results[idx].index = idx;
    results[idx].score = similarity;
}
