# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:2A202603864**
- **Họ và tên: NGUYỄN VĂN QUỐC**
- **SEED: 4842** 

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.910 | 0.857 | 0.085 | 0 |
| 2. BoW + MLP | 0.950 | 0.950 | 0.228 | 0 |
| 3a. Encoder + kNN | 0.902 | 0.923 | 64.7 | 0 |
| 3b. Encoder + LogReg | 0.936 | 0.923 | 39.8 | 0 |
| 3c. Encoder + MLP head | 0.956 | 0.976 | 58.7 | 0 |
| 4a. LLM nhỏ zero-shot | — | 0.976 | 1141.7 | 0.008 |
| 4b. LLM lớn zero-shot | — | 0.976 | 1241.3 | 0.077 |
| 4c. LLM nhỏ few-shot | — | 0.976 | 1286.6 | 0.017 |
## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
1. **lr = 50:** đoán loss dao động răng cưa kéo dài → thực tế vọt lên ~2.6 rồi hội tụ về 0.006 (thấp nhất). Lệch vì loss logistic là hàm lồi, gradient bị chặn, và dữ liệu gần như tách tuyến tính nên lr lớn chỉ làm trọng số tăng nhanh.
2. **Encoder vs BoW trên 6 tin "tricky":** đoán Encoder thắng rõ → thực tế hoà 5/6, cùng sai tin lừa đảo tài khoản. Lệch vì các tin ham có đủ từ trò chuyện để BoW đoán đúng, và kiểu phishing không có trong dữ liệu train nên bộ phân loại phía sau Encoder cũng không nhận ra.
3. **Prompt injection:** đoán đây là nhóm khó nhất với LLM → thực tế LLM đúng 100% cả khi chưa có prompt phòng thủ. Lệch vì model hiện đại đã được huấn luyện để chống kiểu tấn công này, và prompt gốc đã tách tin nhắn bằng `"""`.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
BoW tụt nhiều nhất (accuracy 55–59% so với F1 0.91–0.95 trên test thường), nặng nhất ở nhóm **Né từ khoá** (0–17%) và **Thao túng** (20–30%).
- *"C0ngr4tulat10ns! Y0u h4ve w0n…"*: BoW coi mỗi biến thể là một token lạ không có trong vocab nên mất hết tín hiệu spam.
- *"Hi Mum, I dropped my phone… send £200…"*: không có từ khoá spam, toàn từ trò chuyện (*I, my, mum*) kéo về ham.

Encoder sai nhiều ở nhóm **Nói về spam** (25–50%) vì gom câu theo chủ đề: tin kể lại chuyện bị lừa nằm gần spam. LLM đạt 100% ở mọi nhóm.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
Đề #1: đổi ngưỡng quyết định 0.5 → 0.2 và 0.8.
Cách làm: 
Tính xác suất một lần: p_te = predict_proba(X_te, w, b), không train lại.
Thử từng ngưỡng: pred = (p_te >= t).astype(int) với t = 0.2, 0.5, 0.8.
Đo: metrics(y_te, pred) để lấy precision, recall và F1.
| Ngưỡng | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.2 | 0.882 | 0.951 | 0.915 |
| 0.5 | 0.989 | 0.843 | 0.910 |
| 0.8 | 1.000 | 0.598 | 0.748 |
Hạ ngưỡng thì recall tăng, precision giảm; nâng ngưỡng thì ngược lại. Ngân hàng sợ chặn nhầm OTP (lỗi FP) nên chọn **0.8** (không chặn nhầm tin nào), chấp nhận để lọt ~40% spam. Ngưỡng nên được chọn trên tập validation.
## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Chọn **BoW + MLP**: F1 0.95, p95 0.23 ms, ~46.000 tin/giây, miễn phí, một máy dư sức xử lý ~116 tin/giây. LLM vượt SLA (p95 ~1.1 s) và tốn ~$2.400/tháng (model nhỏ). Đánh đổi: BoW yếu với spam né từ khoá và lừa đảo kiểu mới (58% trên 50 case khó). Cải tiến bằng router: tin có p ∈ [0.3, 0.7] chuyển sang LLM xử lý bất đồng bộ.
## 7. AI log và điều đã học
- Nhắc lại các khái niệm và hướng dẫn tạo vectorize, cosine , call_openai, prompt, parse_label, llm_classify_batch) nhận giải thích và hướng dẫn rõ ràng ; tôi hỏi nhiều cách và đọc hiểu những cách AI gợi ý sau đó tự code lại và thử với dữ liệu.
- Hỏi lỗi `max_tokens=3` khiến LLM trả rỗng → tăng lên 16.
- Hỏi gợi ý trả lời một số câu trả lời dự đoán, vì một số câu ở phầm LLm tôi khônghieeur mà không dự đoán thì không chạy được, sau đó tôi kiểm tra lại với kết quả khi cạy code .
- Hỏi AI gợi ý một số test case ở phần cuối cùng. tôi hỏi lại AI để chọn lọc một số testcase tôi thấy có vẻ tốt nhất

Điều tôi hiểu ra: điểm số phụ thuộc rất nhiều vào dữ liệu đánh giá. Trên tập test thường BoW+MLP hoà với LLM, nhưng trên case khó chỉ đạt 58% so với 100%. Chọn mô hình phải cân nhắc cả chất lượng, độ trễ và chi phí, không chỉ F1.
*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
