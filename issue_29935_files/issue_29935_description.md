# perf (CUDA): fattn KV streaming on Ampere (sm86) is ~2x off physics at long context; quantized-KV + batch>1 always pays an f16-conversion pass (#29935)

**Author:** @Xeoph

### Environment

- llama.cpp b11379 (1537a0a8b); also verified the dispatch code is unchanged on master 11fe02151 (2026-10-04; `ggml/src/ggml-cuda/fattn.cu` last touched by ec7630a64, 2026-10-01). Follow-up experiments below were done on a local build of master 836d571 (= b11379 + 2 commits, fattn identical), MSVC 19.44 + CUDA 13.4, sm86-only.
- 2x RTX 3080 20GB (sm86, `-sm tensor`), Windows, model = qwen3.5-arch 27B NVFP4 (17 full-attn layers with GQA 6, head_dim 256; 48 GDN layers), `-fa on` (forced by tensor split), KV `q8_0/q8_0` or `f16/f16`, speculative draft-MTP x3 (verify batch 4).

### Observation

decode tok/s decays with context at ~0.2 ms per 1K KV tokens per step, while vLLM (FA2/FlashInfer, fp8 KV, TP) decays at ~0.085 ms on the same hardware/model — 2.2-2.7x difference. Short-context step time is identical (~47 ms), so the gap is purely the context slope. Draft acceptance stays constant (~0.5, mean len 2.4-2.6), so it is not a draft-quality effect.

Controlled probes (fresh 30K-token prompt, single request, server-side timing):

| config | decode t/s @30K |
|---|---|
| tensor split, q8_0 KV, MTP3 | 43.6 |
| tensor split, f16 KV, MTP3 | 45.1 |
| tensor split, q8_0 KV, no spec | 38.2 |
| layer split, f16 KV, MTP3 | 32.5 |

Same-session two-point runs (fresh 30K and 90K prompts, same protocol, server-side `eval time`):

| build | KV | @30K | @90K | step-time slope |
|---|---|---|---|---|
| official b11379 | f16 | 48.2 | – | – |
| local 836d571 | f16 | 50.6 | 44.0 | ~0.075 ms / 1K tok |
| local 836d571 | q8_0 | 48.7 | 39.4 | ~0.081 ms / 1K tok (conversion pass visible) |

The physics floor for streaming 30K->90K KV (17 layers, GQA 6, D=256, head-split shards, ~26 KB/token/GPU for f16) is ~0.034 ms per 1K tokens per step, so the observed slope is ~2.2x that for f16 KV. Notably **f16 KV (no conversion needed) decays with a clearly smaller slope than q8_0** in the controlled pair — the whole-cache f16 conversion (`need_f16_K/V` for TILE/MMA) adds decay on top of the streaming term.

### Code pointers (`ggml/src/ggml-cuda/fattn.cu` @ b11379, unchanged on master 11fe02151)

- `ggml_cuda_get_best_fattn_kernel`: on sm86 with quantized K/V, the VEC kernel is only chosen for `Q->ne[1] == 1`; any batch > 1 falls to MMA_F16 (sm89+ gets `<= 2`). With MTP3 the verify batch is 4, so quantized KV always takes the MMA path plus the whole-cache f16 conversion.
- TILE/MMA unconditionally set `need_f16_K = need_f16_V = true` (whole-cache f16 conversion per attention call).
- VEC has native templates for Q8_0/Q4_0/F16/BF16 at D = 64/128/256 (`FATTN_VEC_CASES_ALL_D`).

### We tested the obvious local fix: relaxing the VEC batch gate — it loses

On the local build we changed nothing but the dispatch gate (so VEC, which natively consumes quantized KV with no conversion pass, runs at verify batch 4):

| experiment | diff in `ggml_cuda_get_best_fattn_kernel` | @30K | @90K | vs same-session baseline |
|---|---|---|---|---|
| E-A: quantized-KV gate `== 1` -> `<= 4` (sm86 branch) | 2 lines | 39.7 | 30.7 | **-19% / -22%** (vs q8_0 baseline 48.7 / 39.4) |
| E-B: force VEC for any KV type at batch `<= 4` | 2 lines | 45.4 | 34.6 | **-10% / -21%** (vs f16 baseline 50.6 / 44.0) |

So the MMA-vs-VEC choice at batch 4 / GQA 6 / D 256 on sm86 is correct in ggml's favor as-is; the residual gap to physics sits inside the MMA kernel's KV streaming (or its tile/layout choices), not in the gate. An ncu attribution profile will follow as a comment.

### Questions

1. Is the ~2.2x-over-physics f16-KV decode slope on sm86 expected to improve with any planned MMA-kernel work (tile size, K-layout/transposed loads, stream-K for this shape)?
2. Any plan for native-quantized KV tiles in the MMA kernel, or a cheaper partial/incremental conversion (the current whole-cache f16 pass makes q8_0 KV decay ~1.6x faster than f16 at long context)?


---

# Comments

## @Xeoph — 2026-10-04T07:37:41Z

ncu attribution data as promised (Nsight Compute 2026.3.0, CUDA 13.4, RTX 3080 20GB, device 0 only, this time on the local 836d571 build):

Setup: same model/flags as above (f16 KV, tensor split, draft-MTP3 so the verify batch is 4), fresh 30K-token prompt, 400 generated tokens. Profiled with `--launch-skip 1050 --launch-count 12`, which lands mid-decode at ~30K context: 6 consecutive (main kernel, stream-k fixup) pairs from the verify pass. Raw numbers per launch:

| kernel | grid | duration | DRAM read | DRAM write | L2 hit | SM occupancy |
|---|---|---|---|---|---|---|
| `flash_attn_ext_f16<256, 256, 8, 8>` (main, stream-k) | (136,1,1) | **2.43 ms** | **~1360 MB** | ~13 MB | 48.5% | **13.7%** |
| `flash_attn_stream_k_fixup_general` | (136,8,8) | 33 us | ~10.6 MB | ~4 MB | 29% | 86% |

Interpretation:

- One layer's KV working set at 30K on one GPU (GQA 6 head-split across 2 GPUs, D=256, f16) is ~92 MB. The main kernel reads **~1360 MB from DRAM per launch — ~15x the KV working set**, consistent with re-streaming the KV per output tile (batch 4 x GQA groups ≈ 16 tiles, partially absorbed by the 48.5% L2 hit rate).
- 17 full-attn layers x 2.43 ms ≈ 41 ms, i.e. fattn alone accounts for ~83% of the ~50 ms decode step at 30K. The fixup kernels are negligible.
- Occupancy is 13.7% on the main kernel; effective DRAM throughput during it is ~560 GB/s of the card's ~760 GB/s. The kernel is bandwidth-bound on *redundant* KV traffic, not latency-bound and not at the DRAM limit.

One observation we can't fully explain yet: the ~15x DRAM amplification at 30K over-predicts the measured 30K->90K slope (step time only grows ~0.075 ms per 1K tokens, far less than 16x the ~0.005 ms/1K-tokens-per-layer physics floor), so the redundancy term apparently scales sub-linearly with context. A profile sweep across 30K/60K/90K would pin that down — happy to run it on request.

Artifacts: `ncu --import` of the report yields the per-launch table above (12 launches, metrics: `gpu__time_duration.sum, dram__bytes_read.sum, dram__bytes_write.sum, lts__t_sector_hit_rate.pct, sm__warps_active.avg.pct_of_peak_sustained_active`).


