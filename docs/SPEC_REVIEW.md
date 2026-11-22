# Đánh giá đặc tả `overview.md` và cách triển khai

Tài liệu này đối chiếu từng mục của đặc tả gốc ([`reference/overview.md`](reference/overview.md))
với bài báo định hướng ([Hegde & Usmani, Stanford 2016](https://hgpu.org/?p=16046); bài báo không được đưa vào repo vì bản quyền)
và ghi lại quyết định: **giữ nguyên**, **điều chỉnh**, hay **bỏ qua / rút gọn**. Lý do được
nêu rõ để nhóm có thể xem lại khi viết báo cáo.

Ký hiệu: ✅ làm đúng như đặc tả · 🔧 làm nhưng có điều chỉnh · ✂️ rút gọn / bỏ qua.

## Tóm tắt các điều chỉnh lớn

1. **Bổ sung GPU (CUDA).** Đặc tả chỉ nói "CPU và MPI", nhưng bài báo §3 là thực nghiệm
   CPU vs GPU, tên project là *gpu-parallel-computing* và máy đích có RTX 2060. Vì vậy có
   thêm backend CUDA (naive / tiled / cuBLAS) cho mọi phép tính, và MPI chạy được với
   tensor trên GPU (staging qua host, không yêu cầu CUDA-aware MPI).
2. **ADMM được cài đặt đúng nghĩa.** Algorithm 3 trong bài báo (*ADMM.SGD*) thực chất là
   parameter server đồng bộ: lấy trung bình tham số, không có biến đối ngẫu và hạng phạt ρ.
   Code cài đặt **consensus ADMM chuẩn** (Boyd 2011, §7.1): giải xấp xỉ bài toán con bằng
   SGD cục bộ, rồi một allreduce cho mỗi vòng. Biến thể "parameter server đồng bộ" của bài
   báo vẫn đo được qua `sync` với `--allreduce naive`, mẫu truyền thông y hệt.
3. **Một máy duy nhất.** i5-12600K có 10 lõi vật lý, nên mọi cấu hình scaling chạy 1–10
   tiến trình trên một node, và MPI đi qua shared memory. Code là MPI chuẩn nên chạy được
   multi-node, nhưng cấu hình mặc định không dùng. Mô hình α-β được hiệu chỉnh cho
   truyền thông nội node.
4. **Kiểm thử gọn.** Theo yêu cầu "không cần test chi tiết", không viết unit test cho từng
   hàm. Thay vào đó có một chương trình `pdl_check` kiểm tra các tính chất quan trọng
   (gradient, tương đương tuần tự ↔ phân tán, đồng bộ replica, CUDA ↔ CPU), chạy qua `ctest`.

## Đối chiếu từng mục

| Mục đặc tả | Quyết định | Triển khai / lý do |
|---|---|---|
| §1 Overview | ✅ | Hệ thống benchmark tự cài đặt, đo hiệu năng, bottleneck, tái lập. |
| §2 Mục tiêu "CPU và MPI" | 🔧 | Thêm GPU (xem điều chỉnh 1). Đánh giá AI agent: xem §6. |
| §3.1 NN bằng C | ✅ | C11 + CUDA C. Có conv, dense, ReLU, max-pool, softmax-CE (công thức (2) của bài báo), SGD momentum/weight decay. Kiểm tra đúng bằng gradient check. |
| §3.2 Sequential baseline | ✅ | `--strategy sequential`; cùng mã với sync nên phép so sánh công bằng. |
| §3.3 MPI training | ✅ | Chia dữ liệu `global`/`iid`/`sorted`, allreduce gradient, kiểm tra replica bằng hash (`--check-sync`). |
| §3.4 Sync SGD | ✅ | `sync`: flat hoặc từng layer, có overlap comm/compute. |
| §3.4 Parallel SGD | 🔧 | Zinkevich one-shot (`--avg-period -1`) và lấy trung bình định kỳ. ShuffleSGD (Algorithm 2) dùng all-to-all với **bin ngẫu nhiên cân bằng** thay cho bước lấy mẫu + k-way merge: chính bài báo chứng minh bước đó có chi phí không đáng kể so với O(D), còn all-to-all, phần chi phối, được đo đầy đủ. |
| §3.4 ADMM | 🔧 | Consensus ADMM chuẩn (điều chỉnh 2). |
| §3.4 Downpour / async | 🔧 | Parameter server ở rank 0, đẩy gradient mỗi n_push bước và lấy tham số mỗi n_fetch bước, không chặn (Isend/Irecv), có đo staleness. **Bỏ**: Adagrad và chia tham số trên nhiều shard server (Dean 2012). Với một node, nhiều shard không thay đổi bản chất phép đo. |
| §4.1 Computation benchmark | ✅ | `pdl_bench_compute`: GEMM, FC, conv fwd/bwd, train-step; 8 cài đặt; quét batch và kích thước model. |
| §4.2 Communication benchmark | ✅ | `pdl_bench_comm`: ping-pong, barrier, 6 thuật toán allreduce, bcast, alltoall; quét message size × số tiến trình. |
| §4.3 Training benchmark | ✅ | 10 phase timer cho mỗi rank; throughput, speedup, efficiency, tỉ lệ comm, load imbalance, hội tụ theo epoch, time-to-accuracy. |
| §4.4 Scalability | ✅ / 🔧 | Strong, weak, batch, model size, process count đều có config. Giới hạn ở 1 node (điều chỉnh 3). |
| §5 Performance analysis | ✅ | `pdlbench aggregate/plot/perfmodel/report`. So sánh với mô hình Hockney α-β và mô hình chi phí của bài báo, kể cả điểm bão hòa Downpour k* = T/τ. |
| §6 Đánh giá AI coding agent | ✂️ | Có giao thức, rubric, mẫu nhật ký effort và công cụ `pdlbench ai-eval` đo LOC, độ dài hàm, cảnh báo, complexity ([`AI_AGENT_EVALUATION.md`](AI_AGENT_EVALUATION.md)). Chính repo này là đối tượng AI-generated. **Phần so sánh với implementation của nhóm** chỉ có khung, vì cần code của nhóm. |
| §7 Experiment management | ✅ | YAML → ma trận chạy × repeats × seed, resume, manifest kèm môi trường, symlink `latest`. |
| §8 Visualization | ✅ | Đủ 11 loại biểu đồ trong đặc tả, xuất PNG và PDF, bảng LaTeX, báo cáo Markdown. |
| §9 Correctness | 🔧 | `pdl_check` + `ctest` (điều chỉnh 4). Có phát hiện NaN/Inf. Môi trường ghi vào mọi file kết quả. |
| §10 Deliverables 1–9, 12 | ✅ | Đều có trong repo. |
| §10 Deliverables 10–11 | ✂️ | *Tập kết quả* và *báo cáo hiệu năng* phải sinh trên máy đích (`scripts/run_all.sh`). Repo chỉ cung cấp pipeline, không có kết quả giả. |
| §11 Research questions | ✅ | Mỗi câu hỏi tương ứng với một config (bảng trong README). |
| §12 Implementation planning | ✂️ | Chỉ giữ phần có giá trị lâu dài: kiến trúc, quyết định thiết kế, rủi ro ([`ARCHITECTURE.md`](ARCHITECTURE.md)) và milestone ([`ROADMAP.md`](ROADMAP.md)). Bỏ phần lập kế hoạch dạng văn bản dài, vì nó không ảnh hưởng tới code. |

## Những gì bài báo có nhưng không làm

* **Model parallelism** (§2, §6). Bài báo chỉ nêu ý tưởng. Mọi model dùng ở đây đều vừa
  một máy và một GPU, nên data parallelism là đối tượng chính, đúng như bài báo kết luận.
* **Phân tích asymptotic trên cụm nhiều máy với băng thông B, độ trễ L.** Các công thức được
  đưa vào [`ALGORITHMS.md`](ALGORITHMS.md) và `perfmodel`, nhưng α và β đo trên một node.
* **Truly asynchronous backprop** (cuối §5.2.1). Bài báo chỉ đề xuất, không có thuật toán cụ thể.

## Những gì thêm vào ngoài đặc tả

* Dữ liệu synthetic tái lập theo `(seed, index)`, dùng cho weak scaling và cấu hình
  224×224 / 40 lớp của bài báo mà không cần dataset gốc.
* Preset `paper`: lớp conv đầu tiên 96 × 11×11 stride 4 cho ra 54×54, khớp §3.1.
* Kernel CUDA tất định (col2im và max-pool backward dạng gather, không dùng atomics) để
  kết quả tái lập được.
* Phân vùng `sorted` (shard sắp theo nhãn, non-IID) để kiểm tra giả thiết của Parallel SGD
  trong §5.1.1.
