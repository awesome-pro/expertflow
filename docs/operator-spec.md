# Operator specification (v1, draft — frozen at Phase 1)

The contract every kernel and every baseline adapter must satisfy. Values below are verified against
`Qwen/Qwen3-30B-A3B-FP8` `config.json` and `model.safetensors.index.json` on 2026-11-09.

## 1. Entry point (identical for candidate and baseline)

```python
def moe_decode(
    x: torch.Tensor,            # [T, H]    bfloat16, contiguous
    topk_ids: torch.Tensor,     # [T, K]    int32, values in [0, E)
    topk_weights: torch.Tensor, # [T, K]    float32, already normalized over K
    w13: torch.Tensor,          # [E, 2I, H] float8_e4m3fn
    w13_scale: torch.Tensor,    # [E, 2I/128, H/128] float32
    w2: torch.Tensor,           # [E, H, I]  float8_e4m3fn
    w2_scale: torch.Tensor,     # [E, H/128, I/128] float32
) -> torch.Tensor:              # [T, H]    bfloat16
```

Shapes: `H=2048`, `I=768`, `E=128`, `K=8`, `T ∈ {1,2,4}` (primary), `T=8` (regression only).

## 2. Semantics

For token `t`, selected expert `e = topk_ids[t, k]`:

```text
gate = x[t] @ W_gate[e]                # [I]
up   = x[t] @ W_up[e]                  # [I]
h    = SiLU(gate) * up                 # [I],  SiLU(v) = v * sigmoid(v)
z    = h @ W_down[e]                   # [H]
y[t] = Σ_k topk_weights[t, k] * z_k    # [H]
```

with FP8 W8A8 arithmetic: activations dynamically quantized to E4M3 per token per 128-element group;
weights dequantized as `W = fp8 * scale` with 128×128 block scales; accumulation in FP32.

## 3. Included in the timed region

1. Dynamic quantization of `x` to E4M3 with per-(token, 128-block) scales.
2. Token→expert dispatch, histogram, alignment/padding, and any workspace scatter the backend performs.
3. gate+up grouped GEMM with 128×128 block scales (real FP8 Tensor Core MMA).
4. SwiGLU epilogue `SiLU(gate)*up`.
5. down grouped GEMM with 128×128 block scales.
6. Weighted combine over the K selections into `y[t]`.
7. FP32→BF16 output cast.

Any intermediate buffer the implementation writes and re-reads on the critical path is included.

## 4. Excluded

Router matmul, softmax, top-k selection, normalization of routing weights, RMSNorm, attention,
embedding, sampling, and any host-side tensor allocation (allocations happen outside the graph).

## 5. Weight layout facts (checkpoint → runtime)

The checkpoint stores **separate** tensors per expert and per projection:

```
model.layers.L.mlp.experts.E.gate_proj.weight         float8_e4m3fn  [I, H]
model.layers.L.mlp.experts.E.gate_proj.weight_scale_inv  float32     [I/128, H/128]
model.layers.L.mlp.experts.E.up_proj.weight           float8_e4m3fn  [I, H]
model.layers.L.mlp.experts.E.up_proj.weight_scale_inv float32       [I/128, H/128]
model.layers.L.mlp.experts.E.down_proj.weight         float8_e4m3fn  [H, I]
model.layers.L.mlp.experts.E.down_proj.weight_scale_inv float32     [H/128, I/128]
```

`weight_scale_inv` is the *inverse* scale (multiply to dequantize), matching vLLM's FP8 convention.
The fused `w13` layout is a **runtime** artifact produced by the loader; the exact fused row order and
any per-token interleaving must be confirmed against the pinned vLLM version (Phase 0, A1).

Quantization config (verbatim): `quant_method: fp8`, `fmt: e4m3`, `activation_scheme: dynamic`,
`weight_block_size: [128, 128]`.

Not quantized (145 modules): `lm_head`, all `input_layernorm`/`post_attention_layernorm`;
`mlp.gate` (router) stays unquantized.

## 6. Reference definitions (for error reporting)

- **R1 float reference:** the §2 math in FP32/BF16 using dequantized weights. Its error vs the FP8
  computation is pure quantization error and is reported separately, never used as the kernel target.
- **R2 quantized reference:** `x` quantized with the exact same formula, weights dequantized in FP32,
  matmuls in FP32. A correct kernel differs from R2 only by Tensor Core arithmetic ordering.
- **R3 baseline:** the pinned vLLM backend called through §1.

Kernel acceptance is against **R2**, and separately reported against **R3**.

## 7. Constraints

- No host synchronization and no memory allocation on the captured hot path (CUDA Graph safe).
- Workspace is preallocated and passed in; kernel entry points take raw pointers + shapes.
- Deterministic output for a fixed input (atomic-order nondeterminism must be bounded and measured;
  prefer deterministic reduction where it costs no measurable latency).
- Supported: `T ∈ {1,2,4,8}`, `H=2048`, `I=768`, `E=128`, `K=8`, `sm_90a`.
