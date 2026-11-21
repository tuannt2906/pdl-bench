# Project Specification
# Deep Learning Song Song và Phân Tán

## 1. Project Overview

Xây dựng một hệ thống benchmark phục vụ nghiên cứu và đánh giá hiệu năng huấn luyện Deep Learning song song và phân tán.

Hệ thống tập trung vào việc tự triển khai quá trình huấn luyện neural network, nghiên cứu các phương pháp phân chia dữ liệu, tính toán gradient, đồng bộ tham số và truyền thông giữa các process.

Project cần có khả năng đánh giá hiệu năng, phân tích bottleneck, so sánh các phương pháp huấn luyện và tái lập kết quả thực nghiệm.

## 2. Research Objectives

Các mục tiêu nghiên cứu chính:

- Đánh giá hiệu năng huấn luyện Deep Learning trên CPU và môi trường MPI.
- Nghiên cứu ảnh hưởng của số process đến thời gian huấn luyện.
- Phân tích chi phí computation, communication và synchronization.
- Đánh giá khả năng scalability của các phương pháp parallel training.
- So sánh hiệu năng và chất lượng hội tụ giữa các phương pháp.
- Xây dựng benchmark có tính reproducibility.
- Đánh giá khả năng sử dụng AI coding agent để xây dựng implementation HPC.

## 3. Core Requirements

### 3.1 Neural Network Training

- Tự triển khai neural network bằng C.
- Hỗ trợ forward propagation.
- Hỗ trợ backward propagation.
- Hỗ trợ loss computation.
- Hỗ trợ gradient computation.
- Hỗ trợ optimizer và weight update.
- Có khả năng huấn luyện và đánh giá model.
- Có cơ chế kiểm tra tính đúng đắn của kết quả.

### 3.2 Sequential Baseline

- Xây dựng implementation chạy tuần tự.
- Sử dụng làm baseline cho các phép đo hiệu năng.
- Có khả năng đo thời gian computation và training.
- Đảm bảo kết quả có thể dùng để đối chiếu với implementation song song.

### 3.3 Parallel and Distributed Training

- Xây dựng implementation sử dụng MPI.
- Hỗ trợ phân chia workload giữa các process.
- Hỗ trợ distributed gradient computation.
- Hỗ trợ gradient synchronization.
- Đảm bảo các process duy trì trạng thái model nhất quán khi sử dụng synchronous training.
- Có khả năng mở rộng số process.
- Hỗ trợ đánh giá trên nhiều cấu hình distributed training.

### 3.4 Distributed Optimization

Nghiên cứu và triển khai các phương pháp:

- Synchronous SGD.
- Parallel SGD.
- ADMM.
- Downpour SGD hoặc asynchronous SGD.

Các phương pháp cần được thiết kế thành các implementation có thể benchmark và so sánh độc lập.

## 4. Benchmark Requirements

### 4.1 Computation Benchmark

- Đánh giá hiệu năng các phép tính neural network.
- Đánh giá ảnh hưởng của kích thước model.
- Đánh giá ảnh hưởng của batch size.
- So sánh các implementation computation khác nhau.

### 4.2 Communication Benchmark

- Đánh giá hiệu năng truyền thông MPI.
- Đánh giá chi phí gradient synchronization.
- Đánh giá ảnh hưởng của message size.
- Đánh giá ảnh hưởng của số process.
- So sánh các phương pháp collective communication.

### 4.3 Training Benchmark

- Đánh giá thời gian huấn luyện tổng thể.
- Đánh giá thời gian từng phase của training.
- Đánh giá throughput.
- Đánh giá speedup.
- Đánh giá parallel efficiency.
- Đánh giá communication overhead.
- Đánh giá convergence performance.

### 4.4 Scalability Benchmark

Hệ thống cần hỗ trợ:

- Strong scaling.
- Weak scaling.
- Batch size scaling.
- Model size scaling.
- Process count scaling.

## 5. Performance Analysis

Xây dựng hệ thống phân tích kết quả benchmark, bao gồm:

- So sánh execution time.
- So sánh speedup và parallel efficiency.
- Phân tích computation và communication time.
- Phân tích synchronization overhead.
- Xác định performance bottleneck.
- Đánh giá scalability.
- Phân tích ảnh hưởng của workload đến hiệu năng.
- Đối chiếu kết quả thực nghiệm với performance model.

## 6. AI Coding Agent Evaluation

Xây dựng một nhánh đánh giá implementation được tạo bởi AI coding agent, sử dụng Claude Code làm đối tượng nghiên cứu.

Các yêu cầu:

- Xây dựng specification thống nhất cho implementation.
- Đánh giá correctness của code được sinh.
- Đánh giá khả năng triển khai MPI.
- Đánh giá khả năng xử lý synchronization và communication.
- Đánh giá runtime performance.
- Đánh giá scalability.
- Đánh giá code quality và maintainability.
- Đánh giá effort cần thiết để hoàn thiện implementation.
- So sánh implementation AI-generated với implementation do nhóm phát triển.

## 7. Experiment Management

Xây dựng hệ thống quản lý thực nghiệm có khả năng:

- Khai báo experiment configuration.
- Quản lý các tham số benchmark.
- Tự động chạy nhiều experiment.
- Lưu kết quả và metadata.
- Quản lý random seed.
- Tái lập experiment.
- Tổng hợp kết quả từ nhiều lần chạy.
- Xuất dữ liệu phục vụ phân tích và trực quan hóa.

## 8. Visualization and Reporting

Hệ thống cần hỗ trợ tạo các báo cáo và biểu đồ:

- Training time comparison.
- Speedup comparison.
- Parallel efficiency.
- Throughput comparison.
- Communication overhead.
- Strong scaling.
- Weak scaling.
- Convergence comparison.
- Model size sensitivity.
- Batch size sensitivity.
- Communication performance comparison.

Kết quả cần có khả năng xuất ra các định dạng phù hợp cho báo cáo nghiên cứu.

## 9. Correctness and Reproducibility

- Có bộ kiểm thử cho các thành phần chính.
- Có khả năng kiểm tra gradient.
- Có khả năng kiểm tra tính nhất quán giữa sequential và distributed implementation.
- Có khả năng kiểm tra model synchronization.
- Có khả năng phát hiện lỗi numerical computation.
- Có cơ chế ghi nhận cấu hình môi trường.
- Có khả năng tái lập các experiment.
- Đảm bảo benchmark không đánh đổi correctness để lấy performance.

## 10. Expected Deliverables

Các sản phẩm cần hoàn thành:

1. Source code của hệ thống training và benchmark.
2. Sequential neural network implementation.
3. MPI distributed training implementation.
4. Các implementation distributed optimization được lựa chọn.
5. Benchmark suite.
6. Experiment configuration và execution scripts.
7. Dataset preparation pipeline.
8. Automated result collection.
9. Performance analysis và visualization tools.
10. Tập hợp kết quả thực nghiệm có thể tái lập.
11. Báo cáo đánh giá hiệu năng và scalability.
12. Tài liệu hướng dẫn build, run và reproduce experiments.

## 11. Research Questions

Project cần có khả năng trả lời các câu hỏi:

- Distributed training có cải thiện hiệu năng so với sequential training không?
- Khi tăng số process, speedup và efficiency thay đổi như thế nào?
- Communication overhead ảnh hưởng đến scalability ra sao?
- Workload và model size ảnh hưởng đến hiệu quả parallelization như thế nào?
- Các phương pháp distributed optimization khác nhau có trade-off gì?
- Các phương pháp communication optimization có cải thiện training performance không?
- AI coding agent có thể tạo ra implementation MPI đúng và hiệu quả đến mức nào?

## 12. Implementation Planning

Claude Code cần tự thực hiện các công việc sau trước khi bắt đầu implementation:

- Phân tích toàn bộ project requirements.
- Xác định phạm vi implementation phù hợp.
- Đề xuất kiến trúc hệ thống.
- Xác định module và trách nhiệm từng module.
- Đề xuất technology stack và dependencies.
- Xác định các design decisions quan trọng.
- Xác định các rủi ro kỹ thuật.
- Xây dựng implementation roadmap.
- Xây dựng testing strategy.
- Xây dựng benchmark methodology.
- Xác định các milestone và tiêu chí hoàn thành.

Ưu tiên thiết kế hệ thống có tính modularity, correctness, reproducibility, extensibility và performance measurement.

Không giới hạn implementation vào các ví dụ minh họa đơn giản. Hãy chủ động đề xuất thiết kế phù hợp với mục tiêu nghiên cứu và khả năng mở rộng của project.