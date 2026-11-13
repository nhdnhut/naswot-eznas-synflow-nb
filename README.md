# naswot-eznas-synflow-nb
> Notebook này để ghi chép lại quá trình tìm hiểu 3 Paper: EZNAS, NASWOT, SynFlow, ...; các nghiên cứu liên quan đến Zero-Cost NAS và các Thư viện liên quan sẽ ứng dụng trong quá trình học và thực hiện. Ngôn ngữ được sử dụng note: Tiếng Việt (Có thể sẽ có phiên bản Tiếng Anh để phục vụ cho việc chuyên sâu ở chuyên ngành.)
> Các paper đã được đính kèm tại /src/papers/... với các mã paper đính kèm - phục vụ cho quá trình đọc và học hỏi từ các paper.
## Tiếp cận
Định hướng tiếp cận của em đối với 3 paper này để hiểu chủ yếu về các quy trình: Bài toán -> lý thuyết -> công thức toán -> thuật toán -> triển khai -> đánh giá. Cụ thể, quy trình của em được diễn ra như sau:
### 1. Ý tưởng bài toán:
- Paper đang giải quyết vấn đề gì?
- Tại sao được nghiên cứu?
- Hạn chế trước của paper ban đầu là gì?
- Ý tưởng cốt lõi mà bài toán đề xuất?
- Outcome của ý tưởng/nghiên cứu đó là gì?
### 2. Lý thuyết toán:
> Do hiện tại, em đang bổ sung - các kiến thức về toán chuyên ngành; nên là em dành khá nhiều thời gian trong việc hiểu và phân tích toán. Do đó, một số phần công thức toán em chỉ dùng các trường hợp thực tế để đưa ra gợi ý và giải thích hoạt động. Em sẽ cố gắng học hỏi thêm sau quá trình này. Các vấn đề được em đặt ra ở cơ sở này bao gồm:
- Khái niệm toán đang dùng?
- Cơ sở hình thành?
- Ý nghĩa phép toán, trực quang hóa hình ảnh?
- Liên hệ giữa các công thức liên quan?
- Biểu thức toán -> Thuật toán?
### 3. Quy trình hoạt động:
- Đầu vào? Đầu ra?
- Khởi tạo gì?
- Cơ sở đầu vào?
- Forward pass/Backward pass thực hiện những gì?
- Score được tính? Đánh giá cấu trúc?
### 4. Triển khai code:
> Chủ yếu, quy trình triển khai code này, em sẽ sừ dụng trên Colab - GPU T4 hoặc máy cá nhân của mình; nên có thể đánh giá trong phạm vi cho phép.
- Cấu trúc code (Đi cùng với thuật toán để đặt đầu ra)?
- Các module, hàm quan trọng được sử dụng trong quá trình phát triển?
- Kích thước và giá trị các biến lẻ tensors?
- Tính chính xác của quá trình tính?
### 5. Đánh giá và nhận xét:
> Em chưa thể có nhiều kết luận đúng cho các phương pháp này, tuy nhiên - các câu hỏi và các vấn đề em thực nghiệm, em sẽ dùng tính các score trên các nền kiến trúc khác nhau; đồng thời là kết luận nhỏ do em rút ra khi quan sát về các điểm mạnh, hạn chế của các phương pháp. Vì vậy, các vấn đề em sẽ đặt ra là:
- Thứ hạng giữa các phương pháp theo score (Có thể xem xét các vấn đề phụ thuộc vào input, batch size, ...)
- Điểm mạnh, hạn chế của các quy trìnhp
- Những vấn đề đặt ra, chưa thể giải quyết?
- Liên quan giữa các nghiên cứu với nhau?
- Đề xuất -> Ở hướng này, em chỉ đặt ra một vấn đề nhỏ, sau khi đã tham khảo qua nhiều paper khác nhau và tìm các cốt lõi cần thiết trong quá trình phát triển.
## Các paper hiện tại em đang nghiên cứu và đọc qua:

| Paper | Năm | Nội dung | Tìm hiểu|
|---|---:|---|---|
| NASWOT | 2021 | activation-based zero-cost architecture scoring | Sử dụng activation overlap và kernel matrix để đánh giá kiến trúc |
| EZNAS | 2022 | evolutionary zero-cost proxy discovery | Cách genetic programming tự động tìm kiếm proxy |
| SynFlow | 2020 | pruning at initialization | synaptic Flow, saliency and iterative pruning |
| ... | ... | ... | ... |

**Neural Architecture Search without Training (NASWOT)**

> arXiv: `2006.04647`;
> Publication: ICML 2021
> Authors: Joseph Mellor, Jack Turner, Amos Storkey, Elliot J. Crowley.
> Paper: https://arxiv.org/abs/2006.04647
> Official code: https://github.com/BayesWatch/nas-without-training

Các nội dung em tập trung tìm hiểu gồm có:

- Activation Patterns 
- Binary Activation Matrix
- Hamming Distance
- Kernel Matrix
- Log-determinant
- Architecture Scoring

**Evolving Zero-Cost Proxies for Neural Architecture Scoring (EZNAS)**

> arXiv: `2209.07413`
> Publication: NeurIPS 2022.
> Authors: Yash Akhauri, J. Pablo Muñoz, Nilesh Jain, Ravi Iyer.
> Paper: https://arxiv.org/abs/2209.07413
> Official code: Do bài này tập trung vào phương pháp đánh giá, nên em chỉ dùng suy luận để định hướng đánh giá paper

Các nội dung em tập trung tìm hiểu gồm có:

- Zero-Cost Proxy Discovery
- Genetic Programming
- Expression Tree
- Tensor Collection
- Mathematical Operators
- Fitness Function
- Kendall's Tau
- Crossover và Mutation
- EZNAS-A
- Generalization across architecture search spaces

**Pruning Neural Networks without Any Data by Iteratively Conserving Synaptic Flow (SynFlow)**

> ArXiv: `2006.05467`
> Publication: NeurIPS 2020.
> Authors: Hidenori Tanaka, Daniel Kunin, Daniel L. K. Yamins, Surya Ganguli.
> Paper: https://arxiv.org/abs/2006.05467
> Official code: https://github.com/ganguli-lab/Synaptic-Flow

Các nội dung em tập trung tìm hiểu gồm có (Bài này là phần tốn thời gian của em nhiều nhất trong quá trình làm):

- Network Pruning.
- Pruning at Initialization.
- Weight Linearization.
- Synaptic Flow.
- Gradient-based Saliency.
- Iterative Pruning.
- Layer Collapse.
- Data-free Scoring.

**Ngoài ra, còn có một số paper liên quan mà em đọc qua bao gồm**

> Đây là các paper liên quan trong quá trình nghiên cứu 3 paper được giao; nên em không đi sâu vào chi tiết các paper - chỉ nắm các phần chính cho các paper mà em đang nghiên cứu. 

| Paper | Năm | Liên quan | Các phần liên quan |
|---|---|---|---|
| Zero-Cost Proxies for Lightweight NAS | 2021 | Proxy evaluation | SNIP, GraSP, SynFlow và cách đánh giá các proxy |
| NAS-Bench-Suite-Zero | 2022 | Benchmark | Đánh giá nhiều ZCP trên nhiều tquan |
| Neural Architecture Design and Robustness: A Dataset | 2023 | Robustness benchmark | Architecture-level robustness evaluation |
| CRoZe | 2023 | Robustness-aware ZCP | Feature, parameter và gradient consistency *(Đây là 1/2 paper em quan tâm nhiều nhất khi tìm hiểu các paper liên quan)*|
| An Evaluation of Zero-Cost Proxies | 2025 | Robustness prediction | Khả năng dự đoán robustness bằng nhiều proxy |
| Zero-Cost Proxy for Adversarial Robustness Evaluation | 2025 | Adversarial robustness | Proxy chuyên biệt cho adversarial robustness |
| TRNAS | 2025 | Training-free robust NAS | Training-free adversarial robust architecture search |
| Prediction-powered Neural Architecture Search | 2026 | Proxy-assisted NAS | Kết hợp ZCP với một lượng ground-truth architecture labels hạn chế *(Đây là 2/2 paper em quan tâm nhiều nhất khi tìm hiểu các paper liên quan)* |

**Các paper em đính kèm (/src/papers/*)**
| File PDF | Paper | Liên quan |
|---|---|---|
| `2006.04647v3.pdf` | NASWOT | Paper chính |
| `2209.07413v3.pdf` | EZNAS | Paper chính |
| `2006.05467v3.pdf` | SynFlow | Paper chính|
| `2306.05031v2.pdf` | CRoZe | Paper liên quan |
| `2610.01317v1.pdf` | Prediction-powered NAS | Paper liên quan |
| `2608.19472v3.pdf` | SynFlow: A Multidimensional Diachronic Semantic Analysis Toolkit | Paper chính|