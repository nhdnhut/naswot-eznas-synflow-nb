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

> arXiv: `2006.04647`
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