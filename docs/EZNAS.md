# EZNAS

**Thông tin paper**

| Trường | Thông tin |
|---|---|
| Tên gốc | *EZNAS: Evolving Zero-Cost Proxies for Neural Architecture Scoring* |
| Tác giả | Yash Akhauri, J. Pablo Muñoz, Nilesh Jain, Ravishankar Iyer |
| Năm / Hội nghị | 2022 - NeurIPS 2022 |
| DOI | [10.52202/068431-2208](https://doi.org/10.52202/068431-2208) |
| arXiv | [2209.07413](https://arxiv.org/abs/2209.07413) |
| Paper chính thức | [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2022/hash/c474dc287c6de27fec2981f812c361e1-Abstract-Conference.html) |
| PDF | [NeurIPS PDF](https://papers.nips.cc/paper_files/paper/2022/file/c474dc287c6de27fec2981f812c361e1-Paper-Conference.pdf) |
| Mã nguồn | [IntelLabs/Hardware-Aware-Automated-Machine-Learning - EZNAS](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/tree/main/EZNAS) |

**Zero-Cost Proxy**
> Trong NAS, khi một neural được hình thành - Các quá trình đó thường đi kèm với nhiều các tập architecture liên quan, với mỗi tập architecture sẽ kết hợp cùng nhiều neural khác nhau. Khi xét các rất nhiều các tập neural architecture đó, để nhận xét các architecture nào có tiềm năng tốt thì: Cách trực tiếp là training từng mạng sau đó đo validation acc -> Với những mạng lớn thì gây tốn nhiều chi phí vì nhiều kiến trúc khác nhau.

Khi xét Zero-cost Proxy 