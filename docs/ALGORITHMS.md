# Algorithms and cost models

Notation (as in the paper): k workers, p parameters, D dataset size, N = D/k samples per
worker, B global mini-batch, b = B/k local batch, α latency, β seconds per byte (1/bandwidth),
γ reduction cost per byte, T time to compute one gradient, τ server time per update.

## 1. Training strategies (`--strategy`)

### sequential — baseline
Mini-batch SGD with momentum μ and weight decay λ:
`v ← μ v + ∇L_B(w) + λ w ;  w ← w − η v` with `L_B = (1/B) Σ_i L_i`, `L_i = −f_{y_i} + log Σ_j e^{f_j}` (paper eqs. 1–4).

### sync — synchronous data-parallel SGD (paper §5.1)
Every rank computes the mean gradient over its slice of the global batch; gradients are
summed by allreduce and divided by k, so the update is exactly the sequential update for
batch B (verified by `pdl_check --tests equivalence`).

* `--partition global` (default): one shared shuffled order per epoch, rank r uses slice r of
  each global batch → identical sample sequence for every k (fair strong scaling).
* `--fusion flat|layer`: one allreduce of p floats per step, or one per layer (L latency terms).
* `--overlap 1`: `MPI_Iallreduce` of each layer's gradients is posted as soon as backward has
  produced them (last layer first) and completed before the update.

Cost per step: `T_comp(b) + T_allreduce(4p bytes, k)`.

### psgd — Parallel SGD (Zinkevich et al. 2010; paper §5.1.1, Algorithms 1–2)
Each worker runs SGD on its own shard; models are averaged `w ← (1/k) Σ w_i`:
`--avg-period H` > 0 every H local steps, `0` once per epoch, `-1` only at the end (one-shot,
the original algorithm; intermediate evaluations use a temporary average).

ShuffleSGD (`--reshuffle 1`): after every epoch each worker permutes its shard and sends
1/k of it to every other worker (`MPI_Alltoallv`), so shards are re-mixed. The paper's
sampling + k-way-merge step (cost O(Lk) + O(dkβ log k), negligible per the paper) is replaced
by balanced random bins. The data movement that dominates, O(D) bytes in total, is
measured in the `shuffle` phase. `--partition sorted` reproduces the paper's "data sorted by
label" scenario where shuffling matters.

Cost per averaging round: `H · T_comp(b) + T_allreduce(4p, k)`.
Paper: computation O(p log 1/ε), communication cost O(Nk) + O(pk), time O(p log k).

### admm — consensus ADMM (Boyd et al. 2011 §7.1; paper §5.1.2)
Problem: `min Σ_i f_i(x_i)  s.t. x_i = z`. Scaled form, one round =
`--admm-local-steps L` SGD steps:

```
x_i ← approx argmin f_i(x) + (ρ/2)‖x − z + u_i‖²      (L SGD steps on ∇f_i(x) + ρ(x − z + u_i))
z   ← (1/k) Σ_i (x_i + u_i)                           (one allreduce of p floats)
u_i ← u_i + x_i − z
```
The evaluated model is z; the primal residual `‖x − z‖` is reported. Algorithm 3 of the paper
(*ADMM.SGD*) is really a synchronous parameter server. That pattern can be measured with
`--strategy sync --allreduce naive`.
Paper: computation O(p log k log 1/ε) + O(p log 1/ε), communication cost O(kp log 1/ε),
time O(p log 1/ε log k).

### downpour — asynchronous Downpour SGD (Dean et al. 2012; paper §5.2, Algorithm 4)
Rank 0 is the parameter server; ranks 1..k are workers.

```
worker: every n_fetch steps  send FETCH, post Irecv(params)   — applied when it arrives
        every step           g = ∇f(w_local); w_local −= η g; acc += g
        every n_push steps   Isend(acc, base_version); acc = 0
server: loop  Probe(any) → GRAD: w −= η_s · acc; version++; staleness = version − base
                          FETCH: send (version, w)
                          DONE: retire worker
```
The server's idle time (`sync_wait`) vs busy time (`comm` + `update`) shows saturation:
with k > T/τ workers the server becomes the bottleneck (paper §5.2.1). Bytes per push and
fetch: 4p. Mean and max staleness are reported.

## 2. Allreduce algorithms (`--allreduce`, `pdl_bench_comm`)

n = message bytes. Costs in the Hockney model, with the reduction term γn where it applies:

| algo | rounds | time | notes |
|---|---|---|---|
| `ring` | 2(k−1) | 2(k−1)α + 2((k−1)/k) n β + ((k−1)/k) n γ | bandwidth optimal; reduce-scatter + allgather |
| `rd` (recursive doubling) | ⌈log₂k⌉ (+2 if k not a power of 2) | log₂k (α + nβ + nγ) | latency optimal; full vector every round |
| `tree` (binomial reduce + bcast) | 2⌈log₂k⌉ | 2 log₂k (α + nβ) + log₂k nγ | the paper's "BitTorrent aggregate" pattern |
| `naive` (central driver) | 2(k−1) at root | 2(k−1)(α + nβ) + (k−1) nγ | parameter-server shape; root is the bottleneck |
| `reduce_bcast` | library | ≈ tree | `MPI_Reduce` + `MPI_Bcast` |
| `mpi` | library | ≈ best of the above | `MPI_Allreduce` (OpenMPI / MPICH pick by size) |

All of them leave bitwise-identical results on every rank.

## 3. Performance model (`pdlbench perfmodel`)

1. **Fit α, β** by least squares on the ping-pong one-way time `t(n) = α + β n`
   (`pdl_bench_comm --pingpong 1`). γ is fitted from the residual of large allreduces.
2. **Predict allreduce time** per algorithm with the table above and compare with
   `pdl_bench_comm` measurements (relative error per size / k).
3. **Predict training speedup** of `sync` runs:
   `T_step(k) = T_comp(1) · (b/B) + T_allreduce(4p, k)`, with `T_comp(1)` measured from the
   k = 1 run's compute phases. Predicted vs measured speedup exposes effects outside the model:
   memory-bandwidth contention, stragglers (hybrid cores), MPI progress.
4. **Downpour**: estimate T (worker gradient time) and τ (server update + comm per push) from
   the phase totals; report k* = T/τ.

## 4. Paper §3 reproduction (`pdl_bench_compute`)

* `conv` with `--conv 1:224:224:96:11:4:0` and batches 1…256 → time vs batch size per
  implementation (paper Fig. 1) and the CPU/GPU ratio (Fig. 2). The paper found linear time
  and a ~40× GPU advantage that grows with batch size.
* `fc` with `--fc-batch 1` (matrix–vector) and sizes 256…4096 → O(n²) time (Fig. 3) and the
  CPU/GPU ratio (Fig. 4).
* `gemm` adds a roofline-style view (GFLOP/s) across naive → blocked → OpenMP → CUDA naive →
  tiled → cuBLAS.
