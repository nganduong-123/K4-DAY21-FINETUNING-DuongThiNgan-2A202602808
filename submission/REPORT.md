# Lab 21 — Evaluation Report

**Họ tên**: Dương Thị Ngân<br>
**MSSV**: 2A202602808<br>
**Ngày chạy**: 07/10/2026<br>
**Tier**: `T4` · **Base model**: `unsloth/Qwen3.5-4B` · **GPU thực tế**: Tesla T4 14,6 GB

## 1. Setup

| Mục | Giá trị |
|---|---|
| Dataset | 250 ticket chăm sóc khách hàng, đầu ra JSON 4 trường |
| Train / validation | 225 / 25, seed 42 |
| Tập đánh giá | 50 target + 15 regression; không dùng `EVAL_LIMIT` |
| `max_length` | 1024; p95 đo được là 98 token |
| `MASK_MODE` | `assistant-only` |
| Epoch / optimizer step | 2 epoch / 30 bước cho cả bốn run |

Tôi giữ `max_length=1024` theo cấu hình T4 chuẩn của lab để tất cả run dùng cùng điều kiện và tránh thay đổi thêm một biến trong thí nghiệm. Tuy nhiên, `token_stats.json` cho thấy p95 chỉ 98 và gợi ý 256; vì vậy 1024 là an toàn nhưng dư thừa bộ nhớ. Chat template giữ nguyên khối `<think>`: `template_check.json` có `open_tag_present=true`, `body_present=true`, và verdict `reasoning preserved — safe to train on traces`.

## 2. Bằng chứng loss mask (NB1)

| Mục | Kết quả |
|---|---|
| Token được tính loss | 39 / 94 |
| `supervised_fraction` | 0,4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi bị mask khỏi loss | `true` |

Đoạn đầu được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh",
 "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần system và user đều nằm trong `masked_preview`, còn nhãn JSON của assistant nằm trong `supervised_preview`. Vì vậy model học cách trả lời, không bị tối ưu để chép lại câu hỏi.

## 3. Baseline đóng băng trước khi train (NB2)

| Run | Target | Regression | Format | Latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0,000 | 0,7911 | 0,000 | 3248,8 |
| (b) base + optimized prompt | 0,765 | 0,7911 | 1,000 | 1025,4 |
| (c) LoRA fine-tune | 0,970 | 0,5222 | 1,000 | 1372,6 |

Baseline (b) thật sự mạnh hơn (a): target tăng 0,765 điểm, format tăng từ 0 lên 1 và latency giảm khoảng 68,4%. Tôi không sửa `OPTIMIZED_PROMPT`; hash đã đóng băng là `719e74d3b6232053`. Do baseline được đo trước khi huấn luyện trên đúng 50 target và 15 regression, so sánh với fine-tune không bị rò rỉ hoặc đổi tập sau khi thấy kết quả.

## 4. Giải phẫu ba cấu hình sai (NB4/NB5)

| Run | Vị trí | r | Trainable | LR | Train loss | Target | Train s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32.464.896 | 1e-4 | 0,6265 | 0,970 | 394,6 | 8,78 |
| `attn_only` | q,v | 283 | 32.456.704 | 1e-4 | 0,5375 | 0,970 | 261,9 | 8,79 |
| `wrong_lr` | text-linear | 16 | 32.464.896 | 1e-5 | 1,5702 | 0,000 | 398,3 | 8,78 |
| `qlora` | text-linear 4-bit | 16 | 32.464.896 | 1e-4 | 0,7058 | 0,940 | 466,0 | 3,86 |

### 4.1 — Rank so với vị trí adapter

`attn_only` hòa `correct` trên target với cùng điểm 0,970 dù chỉ gắn vào q,v. Nó còn có train loss thấp hơn, 0,5375 so với 0,6265, nên thứ tự theo loss không mâu thuẫn nhưng cũng không chứng minh được attention-only tổng quát hơn. Kết quả cho thấy trong tác vụ JSON triage hẹp này, tăng rank lên 283 để khớp số tham số có thể bù cho phạm vi gắn adapter hẹp; không thể kết luận “all-linear luôn thắng” chỉ từ lý thuyết. Tôi vẫn ưu tiên `correct` cho thiết kế ban đầu vì rank 16 nhỏ hơn nhiều và cách gắn rộng hơn có cơ sở tốt hơn khi tác vụ đa dạng.

### 4.2 — Learning rate

`wrong_lr` chỉ giảm LR từ 1e-4 xuống 1e-5 nhưng học chậm rõ rệt: sau một epoch loss còn khoảng 1,606, trong khi `correct` đã xuống 0,139. Cuối 30 bước, loss của `wrong_lr` vẫn 1,5702 và target/format đều bằng 0. Nếu chỉ nhìn loss đang giảm mà không biết LR, tôi có thể kết luận sai rằng model chỉ cần train thêm; thực tế cùng ngân sách bước cho thấy LR kiểu full fine-tune không đủ cho LoRA trong thí nghiệm này.

### 4.3 — QLoRA

QLoRA dùng 3,86 GB thay vì 8,78 GB, tiết kiệm 4,92 GB, tương đương khoảng 56,0% peak VRAM. Đổi lại target giảm từ 0,970 xuống 0,940, latency tăng từ 1372,6 lên 1786,0 ms và thời gian train tăng từ 394,6 lên 466,0 giây. Số đo này ủng hộ khuyến nghị không dùng QLoRA cho Qwen3.5 khi T4 vẫn đủ chỗ cho FP16: tiết kiệm bộ nhớ là thật, nhưng không miễn phí về chất lượng và tốc độ.

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy: FAILED**<br>
`target Δ = +0,205` · `regression Δ = -0,269` · `valid_trace_rate = 0,00`

Fine-tune thắng rõ baseline prompt tốt trên tác vụ đích: 0,970 so với 0,765, chênh 0,205; format đều đạt 1,000. Tuy nhiên, regression giảm từ 0,7911 xuống 0,5222, vượt xa ngưỡng cho phép 0,020. Do đó đây không phải một chiến thắng đủ điều kiện triển khai mà là đánh đổi: model chuyên môn hóa tốt hơn cho JSON triage nhưng quên một phần năng lực chung. Tôi giữ nguyên verdict `FAILED`, không nới ngưỡng và không làm yếu baseline. Chẩn đoán hợp lý nhất là catastrophic forgetting do 225 mẫu train chỉ chứa tác vụ triage. Lượt tiếp theo nên trộn 1–5% replay data tổng quát, giữ nguyên tập eval đã đóng băng rồi đo lại cả target và regression.

## 6. Phân tích định tính

NB5 lưu `qualitative.json`; ba ca tệ nhất đều được 0,75 vì dự đoán `urgency=trung_binh` thay vì nhãn đúng `thap`. Ba ca tốt nhất đạt đủ 4/4 trường.

| # | Ticket rút gọn | Nhãn đúng | Fine-tune | Nhận xét |
|---:|---|---|---|---|
| 3 | Bình giữ nhiệt, chưa thấy tiền, khi nào tiện | `hoan_tien/thap/tich_cuc` | `hoan_tien/trung_binh/tich_cuc` | ❌ Sai urgency |
| 5 | Nồi chiên thiếu phụ kiện, khi nào tiện | `san_pham_loi/thap/trung_tinh` | `san_pham_loi/trung_binh/trung_tinh` | ❌ Sai urgency |
| 12 | Áo khoác bị lỗi, khi nào tiện | `san_pham_loi/thap/tich_cuc` | `san_pham_loi/trung_binh/tich_cuc` | ❌ Sai urgency |
| 47 | Ốp lưng, shipper không gọi, hỏi cho biết | `van_chuyen/thap/tich_cuc` | Đúng 4/4 trường | ✅ Fine-tune đúng |
| 48 | Ốp lưng, hỏi giá | `hoi_thong_tin/trung_binh/trung_tinh` | Đúng 4/4 trường | ✅ Fine-tune đúng |
| 49 | Ốp lưng sai màu, sớm nhé | `san_pham_loi/trung_binh/trung_tinh` | Đúng 4/4 trường | ✅ Fine-tune đúng |

Mẫu lỗi chung là các câu giảm mức khẩn cấp bằng cụm “khi nào tiện”. Model nhận đúng intent, product và sentiment nhưng thiên về `trung_binh`, cho thấy ranh giới `thap`/`trung_binh` trong dữ liệu cần thêm ví dụ khó hoặc cân bằng nhãn. Aggregate baseline (b) đã được lưu đầy đủ; notebook gốc không lưu prediction theo từng item của baseline nên tôi không dựng lại câu trả lời per-item không có trong artifact.

## 7. Kết luận và điều học được

Tôi chưa nên deploy adapter này. Nó chứng minh LoRA có thể đưa hành vi phân loại JSON vào trọng số: target tăng 0,205 so với prompt tối ưu và format đạt 100%. Nhưng cổng regression thất bại nặng với mức giảm 0,269, nên lợi ích tác vụ đích chưa bù được rủi ro mất năng lực chung. Đòn bẩy lớn nhất trong lab này là learning rate và chất lượng/phạm vi dữ liệu, không phải chỉ tăng rank. LR thấp hơn 10 lần làm target về 0 dù kiến trúc adapter giống hệt. Ngược lại, attention-only rank 283 hòa all-linear rank 16, nên “vị trí adapter luôn quan trọng hơn rank” không đúng tuyệt đối trên bài toán hẹp này. Mask đúng là điều kiện nền tảng; nếu mask sai thì mọi so sánh sau đó đều mất ý nghĩa. Bước cải thiện hợp lý là thêm replay data tổng quát, tăng ví dụ phân biệt urgency thấp với trung bình, giữ nguyên 50 target và 15 regression, rồi chạy lại đúng cổng hiện tại.

Ba điều tôi học được:

1. Phải đóng băng baseline prompt mạnh trước khi train; nếu chỉ so với prompt ngây thơ thì dễ tuyên bố thắng giả.
2. Train loss không thay cho task metric: `attn_only` có loss thấp hơn nhưng chỉ hòa target, còn `wrong_lr` cho thấy LR có thể phá hỏng cả format.
3. Tiết kiệm VRAM bằng QLoRA phải được định giá bằng target, latency và thời gian train, không chỉ bằng số GB.

Nếu có thêm 2 giờ, tôi sẽ trộn 1%, 3% và 5% replay data, chạy cùng 30 bước, chọn mức nhỏ nhất đưa regression delta về trong -0,020 mà vẫn giữ target trên 0,765.

## Phụ lục

- [ ] NB6 merge + hot-swap
- [ ] Dataset miền riêng
- [ ] So sánh hai `MASK_MODE`
- [ ] Quét rank có kiểm soát
- [ ] Đẩy adapter lên Hugging Face Hub
