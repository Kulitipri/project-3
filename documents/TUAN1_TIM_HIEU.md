# Tuần 1 — Tìm hiểu đề tài, DeepLog và bộ dữ liệu HDFS_v1

Ghi chú lại những gì đã tìm hiểu trong tuần 1. Tuần này chưa xây dựng mô hình. Mục tiêu là đặt bài toán
cho rõ, chốt phạm vi và chọn cách làm cho các tuần sau. Phần code duy nhất của tuần là script khảo sát
bộ mẫu HDFS_2k (`src/explore_hdfs_2k.py`, xem README mục 4).

## 1. Bài toán

Mỗi đơn vị thực thi của hệ thống được gọi là một **phiên**. Với HDFS, một phiên là vòng đời của một
block dữ liệu: mọi dòng log nhắc tới cùng một mã `blk_...` thuộc về phiên của block đó.

- **Đầu vào:** chuỗi sự kiện có thứ tự của một block, ví dụ "cấp phát block → đang nhận block →
  đang nhận block → ... → cập nhật block đã lưu".
- **Đầu ra:** block đó bình thường hay bất thường.
- **Cách học: bán giám sát.** Mô hình chỉ được huấn luyện trên các phiên bình thường. Phiên nào không
  khớp với mẫu bình thường đã học thì bị báo là bất thường. Nhãn của HDFS_v1 chỉ dùng để đánh giá,
  vì trong thực tế việc gán nhãn sự cố rất tốn công.

## 2. Pipeline chung

Theo nghiên cứu [5], các phương pháp phát hiện bất thường trên log đều đi qua các bước giống nhau:

1. **Phân tích cú pháp (parsing):** biến nội dung tự do thành *template* (mã sự kiện) cộng với các biến.
   Ví dụ `Receiving block blk_123 src: /10.0.0.1:5001 dest: /10.0.0.2:50010`
   → `Receiving block <*> src: <*> dest: <*>`. Công cụ: Drain [4], bản Python là Drain3 [10].
2. **Gom nhóm:** gom các sự kiện thuộc cùng một phiên. HDFS dùng mã block làm khóa phiên. Bộ dữ liệu
   nào không có mã định danh thì gom theo cửa sổ thời gian.
3. **Biểu diễn:** dạng chuỗi mã sự kiện (giữ thứ tự) hoặc dạng vector đếm (mất thứ tự).
4. **Phát hiện:** một mô hình học hành vi bình thường từ cách biểu diễn đó.

| Phương pháp | Loại | Ý tưởng | Đầu vào |
|---|---|---|---|
| PCA [2] | Cổ điển | Vector đếm của phiên bình thường nằm trong một không gian con ít chiều. Vector nằm xa không gian con đó là bất thường. | Vector đếm |
| Invariant Mining [11] | Cổ điển | Học quan hệ tuyến tính giữa số lần xuất hiện của các sự kiện (mỗi "open" phải có một "close"). | Vector đếm |
| **DeepLog [1]** | Học sâu (LSTM) | Dự đoán sự kiện tiếp theo. Nếu sự kiện thật nằm ngoài dự đoán thì là bất thường. | Chuỗi sự kiện |
| LogAnomaly [6] | Học sâu (LSTM) | Giống DeepLog, thêm vector đếm và biểu diễn ngữ nghĩa của template. | Chuỗi + vector đếm |
| LogRobust [7] | Học sâu (Bi-LSTM + attention) | Phân loại có giám sát trên vector ngữ nghĩa, chịu được việc log thay đổi. | Vector ngữ nghĩa |

DeepLog được chọn vì bốn lý do: được trích dẫn nhiều, không cần nhãn bất thường khi huấn luyện, đủ đơn
giản để tự cài đặt, và là mốc so sánh của hầu hết các bài báo sau.

## 3. DeepLog — mô hình chuỗi sự kiện (log key model)

DeepLog coi log như một "ngôn ngữ", trong đó mỗi mã sự kiện là một "từ". Mô hình LSTM được huấn luyện trên
các phiên bình thường để dự đoán sự kiện tiếp theo từ `h` sự kiện trước đó. Khi phát hiện:

1. lấy `h` sự kiện gần nhất của phiên;
2. LSTM cho ra xác suất của mọi sự kiện có thể xảy ra tiếp theo;
3. giữ lại `g` sự kiện có xác suất cao nhất;
4. nếu sự kiện thật **không nằm trong top-g** thì đánh dấu phiên là bất thường.

Tham số trong bài báo: `h = 10`, `g = 9`, LSTM 2 tầng, mỗi tầng 64 đơn vị.
Project III chỉ cài đặt mô hình này. Mô hình giá trị tham số và phần dựng workflow trong [1] nằm ngoài phạm vi.

## 4. Kết quả công bố trên HDFS và mục tiêu tái tạo

Thiết lập của bài báo: huấn luyện trên 4.855 phiên bình thường lấy từ 100.000 dòng log đầu (dưới 1% dữ liệu),
kiểm tra trên 553.366 phiên bình thường và 15.200 phiên bất thường.

| Phương pháp | FP | FN |
|---|---|---|
| PCA | 277 | 5.400 |
| Invariant Mining | 2.122 | 1.217 |
| N-gram (N = 1) | 1.360 | 739 |
| **DeepLog** | **833** | **619** |

Từ hai con số của DeepLog:

- TP = 15.200 − 619 = 14.581
- precision = 14.581 / (14.581 + 833) ≈ 0,946
- recall = 14.581 / 15.200 ≈ 0,959
- **F1 ≈ 0,95**. Đây là con số mục tiêu khi tái tạo DeepLog.

## 5. Lưu ý từ nghiên cứu sau này (Le & Zhang, ICSE 2022 [8])

Nghiên cứu này đánh giá lại 5 mô hình học sâu, trong đó có DeepLog, trên 4 bộ dữ liệu. Kết luận chính là
điểm số rất cao trên HDFS phụ thuộc nhiều vào cách thiết lập thí nghiệm: dữ liệu nào được dùng để huấn
luyện, cách gom log, tỉ lệ giữa các lớp, nhiễu nhãn, và việc chia dữ liệu theo thời gian hay ngẫu nhiên.
Từ đó đề tài rút ra hai quy tắc:

1. **Chia dữ liệu theo thứ tự thời gian:** huấn luyện trên các block xuất hiện trước, kiểm tra trên các
   block xuất hiện sau, để mô hình không bao giờ thấy dữ liệu "tương lai".
2. **Mọi kết quả đều ghi rõ cách chia dữ liệu đã dùng.**

## 6. Bộ dữ liệu HDFS_v1 (Loghub [3])

| Thuộc tính | Giá trị |
|---|---|
| Hệ thống nguồn | Hadoop Distributed File System, khoảng 200 node Amazon EC2 |
| Số dòng log | 11.175.629 |
| Khoảng thời gian | 38,7 giờ |
| Dung lượng | khoảng 1,47 GB |
| Số phiên (block ID) | khoảng 575.000 |
| Phiên bất thường | 16.838 (khoảng 2,9%) |
| Số loại sự kiện | 29 (theo bài báo DeepLog) |
| Nhãn | theo từng block, trong `anomaly_label.csv` |

- Log được sinh ra bằng các tác vụ benchmark trong một môi trường đám mây riêng. Nhãn được gán bằng các
  luật viết tay.
- **Dữ liệu mất cân bằng:** một mô hình luôn trả lời "bình thường" đã đạt accuracy khoảng 97%. Vì vậy
  đề tài dùng **precision, recall và F1 trên lớp bất thường**, không dùng accuracy.
- Loghub có sẵn các file đã tiền xử lý (danh sách template, chuỗi sự kiện theo block). Log gốc vẫn được
  tự parse bằng Drain3, vì bước parse là một phần của pipeline. Các file của Loghub chỉ dùng để đối chiếu.

## 7. Khảo sát bộ mẫu HDFS_2k và hệ quả

Chạy `python -m src.explore_hdfs_2k` (cách tải dữ liệu và kết quả chi tiết xem README mục 3–4).

| Phát hiện | Hệ quả cho các tuần sau |
|---|---|
| Cả 2.000 dòng khớp định dạng `date time pid level component: message` | Một regex là đủ để tách phần đầu dòng |
| 1.920 INFO, 80 WARN, không có ERROR. Cả 80 dòng WARN đều là "Got exception while serving block" | Chỉ tìm chữ "ERROR" là không đủ để phát hiện sự cố |
| Drain3 ra 16 template, Loghub có 14. Template "BLOCK* ask ... to delete ..." (E4 trong đáp án của bộ mẫu) bị tách theo số mã block trong dòng | Phải thêm luật che riêng cho trường hợp này trước khi parse HDFS_v1 (tuần 2) |
| 2.200 mã block, 2.194 mã chỉ xuất hiện trong đúng 1 dòng | Bộ mẫu được lấy ngẫu nhiên theo dòng nên không dựng được chuỗi theo block. Huấn luyện và đánh giá phải dùng HDFS_v1 đầy đủ |

## 8. Phạm vi và các quyết định cho các tuần sau

| Trong phạm vi (Project III) | Ngoài phạm vi (hướng đồ án tốt nghiệp) |
|---|---|
| Một bộ dữ liệu: HDFS_v1 | Nhiều nguồn log cùng lúc |
| Chỉ mô hình chuỗi sự kiện của DeepLog | Mô hình giá trị tham số, chẩn đoán workflow |
| Đánh giá ngoại tuyến trên một cách chia cố định | Xử lý luồng thời gian thực, cập nhật mô hình trực tuyến |
| Script phát hiện chạy bằng dòng lệnh | Dashboard, quản lý cảnh báo, mini-SIEM hoàn chỉnh |
| Một phương pháp cổ điển để so sánh | LogAnomaly, LogRobust, mô hình dựa trên transformer |

Các quyết định:

- **Chia dữ liệu theo thời gian.** Thí nghiệm chính dùng thiết lập giống bài báo DeepLog (tập huấn luyện
  nhỏ gồm các phiên bình thường lấy từ đầu log) để so sánh trực tiếp với [1]. Thí nghiệm thứ hai chia
  80/20 theo thời gian để so sánh với [8].
- **Parse bằng Drain3, đối chiếu với Loghub.** Nếu đến cuối tuần 2 kết quả parse vẫn lệch với Loghub,
  mô hình sẽ được huấn luyện trên chuỗi đã tiền xử lý của Loghub để phần mô hình không bị chậm tiến độ.
- **Chỉ làm mô hình chuỗi sự kiện của DeepLog**, vì đây là phần được đánh giá trên HDFS trong [1].
- **Thử nghiệm trên AIT-LDS v2.0 [9] (tuần 4) là phần không bắt buộc.** Phần này chỉ bắt đầu sau khi
  có kết quả trên HDFS.
- **Khó khăn:** chưa quen dùng LSTM trong PyTorch (chia batch cho chuỗi, kích thước đầu vào/đầu ra), nên
  tuần 2 dành một phần thời gian làm bài tập PyTorch nhỏ trước khi cài đặt DeepLog.

## Tài liệu tham khảo

1. M. Du, F. Li, G. Zheng, V. Srikumar. *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning.* ACM CCS 2017.
2. W. Xu, L. Huang, A. Fox, D. Patterson, M. I. Jordan. *Detecting Large-Scale System Problems by Mining Console Logs.* ACM SOSP 2009.
3. J. Zhu, S. He, P. He, J. Liu, M. R. Lyu. *Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics.* IEEE ISSRE 2023. <https://github.com/logpai/loghub>
4. P. He, J. Zhu, Z. Zheng, M. R. Lyu. *Drain: An Online Log Parsing Approach with Fixed Depth Tree.* IEEE ICWS 2017.
5. S. He, J. Zhu, P. He, M. R. Lyu. *Experience Report: System Log Analysis for Anomaly Detection.* IEEE ISSRE 2016.
6. W. Meng et al. *LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in Unstructured Logs.* IJCAI 2019.
7. X. Zhang et al. *Robust Log-based Anomaly Detection on Unstable Log Data.* ESEC/FSE 2019.
8. V.-H. Le, H. Zhang. *Log-based Anomaly Detection with Deep Learning: How Far Are We?* ICSE 2022.
9. M. Landauer et al. *AIT Log Data Set V2.0.* Zenodo. <https://zenodo.org/records/5789064>
10. Drain3: a robust streaming log template miner. <https://github.com/logpai/Drain3>
11. J.-G. Lou, Q. Fu, S. Yang, Y. Xu, J. Li. *Mining Invariants from Console Logs for System Problem Detection.* USENIX ATC 2010.
