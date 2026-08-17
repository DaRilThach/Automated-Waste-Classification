import os
from pathlib import Path

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Cấu hình đường dẫn dữ liệu
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Đường dẫn lưu trữ kết quả và trọng số mô hình
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Danh sách nhãn phân loại (6 lớp chuẩn)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)

# Ánh xạ tên tiếng Việt và hướng dẫn xử lý môi trường
CLASS_METADATA = {
    "cardboard": {
        "vn_name": "Bìa carton",
        "category": "Rác tái chế (Recyclable)",
        "color": "#e67e22",
        "is_recyclable": True,
        "guide": "Làm phẳng hộp carton, giữ khô ráo, tháo bỏ băng dính bẩn trước khi phân loại tái chế.",
        "icon": "📦"
    },
    "glass": {
        "vn_name": "Thủy tinh",
        "category": "Rác tái chế (Recyclable)",
        "color": "#3498db",
        "is_recyclable": True,
        "guide": "Tráng sạch cặn thức ăn hoặc đồ uống, bọc cẩn thận nếu vỡ để tránh gây nguy hiểm cho người thu gom.",
        "icon": "🍾"
    },
    "metal": {
        "vn_name": "Kim loại",
        "category": "Rác tái chế (Recyclable)",
        "color": "#95a5a6",
        "is_recyclable": True,
        "guide": "Rửa sạch vỏ lon (nhôm/sắt), làm bẹp lon để tiết kiệm diện tích lưu trữ tái chế.",
        "icon": "🥫"
    },
    "paper": {
        "vn_name": "Giấy",
        "category": "Rác tái chế (Recyclable)",
        "color": "#f1c40f",
        "is_recyclable": True,
        "guide": "Giữ giấy khô ráo, không lẫn dầu mỡ hay nilon. Giấy bẩn hoặc dính dầu nên bỏ vào rác thông thường.",
        "icon": "📄"
    },
    "plastic": {
        "vn_name": "Nhựa",
        "category": "Rác tái chế (Recyclable)",
        "color": "#e74c3c",
        "is_recyclable": True,
        "guide": "Đổ hết chất lỏng, xúc sạch, tháo nắp chai và ép bẹp chai nhựa trước khi phân loại.",
        "icon": "🧴"
    },
    "trash": {
        "vn_name": "Rác hỗn hợp / Vô cơ khác",
        "category": "Rác không tái chế (Non-recyclable)",
        "color": "#7f8c8d",
        "is_recyclable": False,
        "guide": "Rác không thể tái chế (túi nilon bẩn, gốm sứ vỡ, tã lót...), cần buộc kín và bỏ vào rác thải sinh hoạt thông thường.",
        "icon": "🗑️"
    }
}

# Cấu hình tham số mô hình & huấn luyện
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3
BATCH_SIZE = 32
RANDOM_SEED = 42

# Tỉ lệ phân chia dữ liệu
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Siêu tham số Huấn luyện
PHASE_1_EPOCHS = 12       # Huấn luyện Top Head
PHASE_1_LR = 1e-3
PHASE_2_EPOCHS = 18       # Fine-tuning Backbone
PHASE_2_LR = 1e-5
TOTAL_EPOCHS = PHASE_1_EPOCHS + PHASE_2_EPOCHS
