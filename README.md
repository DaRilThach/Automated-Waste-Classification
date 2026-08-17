# ♻️ Automated Waste Classification (Hệ Thống Phân Loại Rác Thải Tự Động)

> **Academic Project** - Ứng dụng Deep Learning & Transfer Learning trong bài toán nhận diện và phân loại rác thải sinh hoạt theo thời gian thực nhằm bảo vệ môi trường và thúc đẩy tái chế bền vững.

---

## 📌 1. Giới thiệu Đề tài & Mục tiêu

Phân loại rác tại nguồn là mắt xích quan trọng nhất trong chuỗi kinh tế tuần hoàn và xử lý chất thải đô thị. Dự án này xây dựng một giải pháp hoàn chỉnh từ **tiền xử lý dữ liệu**, **huấn luyện mạng nơ-ron học sâu (Deep Learning)**, **đánh giá toàn diện các chỉ số khoa học**, đến **triển khai ứng dụng Web/App thời gian thực**.

### Các mục tiêu cốt lõi:
1. **Thu thập & Tiền xử lý dữ liệu**: Chuẩn hóa tập dữ liệu **Garbage Classification** (Kaggle) quy mô 2.527 hình ảnh, phân tầng (Stratified) thành 6 lớp, áp dụng Data Augmentation đa dạng (xoay, lật, zoom, đổi tương phản).
2. **Huấn luyện & Tối ưu hóa (Fine-tuning)**: Xây dựng và so sánh 2 kiến trúc mạng hàng đầu là **MobileNetV2** (tối ưu hóa suy luận nhẹ, siêu nhanh) và **ResNet50** (trích xuất đặc trưng sâu), áp dụng chiến lược huấn luyện 2 giai đoạn (Transfer Learning + Fine-tuning).
3. **Đánh giá Độc lập**: Đạt độ chính xác **Accuracy > 85%** trên tập test độc lập; đánh giá chi tiết qua **Precision, Recall, F1-score, Confusion Matrix** và biểu đồ huấn luyện.
4. **Triển khai Web Prototype**: Xây dựng giao diện Web tương tác trực quan cho phép phân loại ảnh tải lên và **phân loại qua Webcam thời gian thực**, hiển thị mức độ tự tin, độ trễ xử lý (ms) và hướng dẫn phân loại tái chế chuẩn môi trường.

---

## 📊 2. Tập Dữ Liệu & Phân Bố 6 Lớp

Tập dữ liệu được phân chia theo tỉ lệ **Train (80%) / Validation (10%) / Test (10%)**:

| Phân lớp (`Class`) | Tên tiếng Việt | Nhóm rác | Số lượng mẫu |
| :--- | :--- | :--- | :---: |
| **`cardboard`** | Bìa carton | ♻️ Tái chế | 403 |
| **`glass`** | Thủy tinh | ♻️ Tái chế | 501 |
| **`metal`** | Kim loại (vỏ lon, hộp sắt) | ♻️ Tái chế | 410 |
| **`paper`** | Giấy vụn, sách báo | ♻️ Tái chế | 594 |
| **`plastic`** | Chai nhựa, đồ nhựa | ♻️ Tái chế | 482 |
| **`trash`** | Rác vô cơ / hỗn hợp | 🗑️ Không tái chế | 137 |
| **TỔNG CỘNG** | **6 Phân Lớp** | - | **2.527 ảnh** |

---

## 🧠 3. Kiến Trúc & Chiến Lược Huấn Luyện

Hệ thống áp dụng phương pháp **Two-Stage Transfer Learning**:

```
[Ảnh Đầu Vào (224x224x3)] 
         │
         ▼
[Data Augmentation (Flip, Rotation, Zoom, Contrast)]
         │
         ▼
[Pretrained Backbone (MobileNetV2 / ResNet50)]
         │
         ▼
[GlobalAveragePooling2D] ──► [BatchNormalization] ──► [Dense(256/512, ReLU) + L2] ──► [Dropout(0.4-0.5)]
         │
         ▼
[Dense(6, Softmax Output)] ──► [Xác suất 6 Phân Lớp]
```

- **Giai đoạn 1 (Warmup / Feature Extraction)**: Đóng băng (Freeze) toàn bộ trọng số ImageNet của Base Model, chỉ cập nhật trọng số của Classification Head với Adam Optimizer ($lr = 10^{-3}$).
- **Giai đoạn 2 (Fine-tuning)**: Mở đóng băng (Unfreeze) 25-30 layers sâu nhất của Backbone, huấn luyện tinh chỉnh với Learning Rate nhỏ ($lr = 10^{-5}$), kết hợp `ReduceLROnPlateau` và `EarlyStopping`.

---

## 🚀 4. Cài Đặt & Hướng Dẫn Sử Dụng

### 4.1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 hoặc 3.11:
```bash
pip install -r requirements.txt
```

### 4.2. Chuẩn bị và phân chia tập dữ liệu
Tự động xác thực tính toàn vẹn và phân chia tập dữ liệu:
```bash
python data/prepare_data.py
```

### 4.3. Huấn luyện mô hình
Huấn luyện MobileNetV2 (mặc định) hoặc ResNet50:
```bash
# Huấn luyện MobileNetV2
python train.py --model mobilenet_v2

# Huấn luyện ResNet50
python train.py --model resnet50

# Huấn luyện cả hai mô hình để so sánh
python train.py --model all
```

### 4.4. Đánh giá độc lập trên tập Test
Tạo ma trận nhầm lẫn (Confusion Matrix) và bảng chỉ số chi tiết:
```bash
# Đánh giá MobileNetV2
python evaluate.py --model mobilenet_v2

# Đánh giá ResNet50
python evaluate.py --model resnet50
```

### 4.5. Dự đoán đơn lẻ qua dòng lệnh (CLI)
```bash
python predict.py --image path/to/sample.jpg --model mobilenet_v2
```

### 4.6. Khởi chạy Web Application (Giao diện Streamlit)
```bash
streamlit run app.py
```
*Truy cập trình duyệt tại địa chỉ `http://localhost:8501` để trải nghiệm:*
- Phân loại ảnh kéo thả (Upload).
- Nhận diện trực tiếp qua **Camera Webcam thời gian thực**.
- Xem biểu đồ xác suất Top-3 và độ trễ suy luận (ms).
- Tra cứu cẩm nang phân loại rác và bảo vệ môi trường.

---

## 📁 5. Cấu Trúc Mã Nguồn Dự Án

```
d:/TGMT/
├── data/
│   ├── prepare_data.py          # Script tải, kiểm tra và phân chia dữ liệu train/val/test
│   └── processed/               # Dữ liệu đã phân tầng (train, val, test)
├── src/
│   ├── config.py                # Cấu hình siêu tham số, đường dẫn, nhãn lớp & thông tin tái chế
│   ├── dataset.py               # Pipeline nạp dữ liệu và Data Augmentation (tf.data)
│   ├── models.py                # Định nghĩa kiến trúc MobileNetV2 & ResNet50 Transfer Learning
│   └── utils.py                 # Hàm vẽ Learning Curve, Confusion Matrix, tính Precision/Recall/F1
├── checkpoints/                 # Thư mục lưu trữ trọng số mô hình tối ưu nhất (.keras)
├── outputs/                     # Thư mục lưu biểu đồ training, confusion matrix và metrics JSON
├── notebooks/
│   └── waste_classification_academic_report.ipynb  # Notebook báo cáo học thuật đầy đủ
├── train.py                     # Script thực thi huấn luyện mô hình 2 giai đoạn
├── evaluate.py                  # Script đánh giá trên tập kiểm tra độc lập
├── predict.py                   # Module suy luận và tiện ích dự đoán nhanh qua CLI
├── app.py                       # Giao diện Web phân loại rác thời gian thực (Streamlit)
├── requirements.txt             # Danh sách các thư viện phụ thuộc
└── README.md                    # Tài liệu hướng dẫn & báo cáo dự án
```

---

## 🌿 6. Ý Nghĩa Thực Tiễn & Xử Lý Môi Trường
 
Hệ thống cung cấp hướng dẫn tức thì cho từng loại rác:
- **Cardboard (Bìa carton)**: Làm phẳng hộp, giữ khô ráo, tháo bỏ băng dính bẩn trước khi thu gom tái chế.
- **Glass (Thủy tinh)**: Tráng sạch cặn thức ăn hoặc đồ uống, bọc cẩn thận nếu vỡ để tránh gây nguy hiểm.
- **Metal (Kim loại)**: Rửa sạch vỏ lon nước ngọt/đồ hộp, ép bẹp để tiết kiệm diện tích.
- **Paper (Giấy)**: Giữ phẳng và khô ráo, không lẫn dầu mỡ hay nilon.
- **Plastic (Nhựa)**: Đổ hết chất lỏng, xúc sạch, tháo nắp và ép bẹp chai nhựa trước khi phân loại.
- **Trash (Rác khác)**: Rác không thể tái chế (túi nilon bẩn, gốm sứ vỡ...), buộc kín và đưa vào luồng rác thải sinh hoạt thông thường.
