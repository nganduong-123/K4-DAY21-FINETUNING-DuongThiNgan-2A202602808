# Reflection — Lab 21

## 1. Điều làm tôi ngạc nhiên nhất

LoRA đúng đạt target 0,970, cao hơn prompt tối ưu 0,765, nhưng vẫn bị verdict `FAILED` vì regression giảm 0,269. Tôi từng nghĩ thắng tác vụ đích là đủ; số đo cho thấy một model chuyên môn hóa có thể đồng thời trở nên kém an toàn hơn để triển khai.

## 2. Phần tốn thời gian nhất

Ba run đối chứng NB4 tốn nhiều nhất, khoảng 21,5 phút, chưa kể hơn 11 phút đánh giá NB5. Tôi dự đoán training sẽ chậm, nhưng không nghĩ đối chứng learning-rate thấp lại sinh chậm tới 5268,9 ms mỗi mẫu và mất hơn bốn phút chỉ để chấm target.

## 3. Niềm tin đã thay đổi

Trước lab, tôi tin rằng tăng rank hoặc gắn LoRA vào nhiều lớp hơn gần như chắc chắn sẽ cho điểm tốt hơn. Kết quả attention-only rank 283 hòa correct ở target 0,970 cho thấy kết luận đó phụ thuộc tác vụ và ngân sách tham số. Tôi cũng không còn tin train loss thấp hơn tự động đồng nghĩa model tốt hơn trên dữ liệu chưa thấy.

## 4. Tôi dùng AI assistant thế nào

Tôi dùng AI assistant để đọc yêu cầu, kiểm tra loss mask, vận hành Colab T4, theo dõi log, đối chiếu artifact và soạn báo cáo từ số đo. Chỗ assistant suýt sai là notebook Colab bị cache ở `EVAL_LIMIT=8`; log đã phát hiện kịp và lượt smoke bị dừng trước khi chạy lại full 50/15. Cuối phiên, cơ chế tải file của Colab cũng lỗi, nên các artifact nhỏ được khôi phục từ chính log đã ghi thay vì giả định tải thành công.

## 5. Nếu fine-tune cho khách hàng thật

Bước đầu tiên là viết acceptance gate trước khi train: target metric, format contract, tập regression đại diện và ngưỡng không được tụt. Sau đó tôi mới đóng băng prompt baseline mạnh, kiểm tra mask và thiết kế replay data. Cách này giúp tránh việc có một model “đẹp trên train loss” nhưng không đủ điều kiện triển khai.
