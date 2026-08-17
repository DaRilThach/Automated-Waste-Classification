import os
import sys
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Thiet lap cau hinh Streamlit Page
st.set_page_config(
    page_title="Hệ thống Phân Loại Rác Thải Tự Động (AI Waste Classification)",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm thư mục gốc vào path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.config import (
    CLASSES,
    CLASS_METADATA,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    IMG_SIZE
)
from predict import WastePredictor

# Tùy biến CSS giao diện hiện đại, tinh tế
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .recyclable-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .non-recyclable-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }
    .guide-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 14px 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model(model_name="mobilenet_v2"):
    """Cache mo hinh da load de tang toc do suy luan."""
    try:
        return WastePredictor(model_type=model_name)
    except Exception as e:
        return None

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/recycling--v1.png", width=70)
    st.title("⚙️ Cấu Hình Hệ Thống")
    
    selected_model_type = st.selectbox(
        "🧠 Lựa chọn Mô hình Deep Learning:",
        ["mobilenet_v2", "resnet50"],
        format_func=lambda x: "MobileNetV2 (Tối ưu Real-Time)" if x == "mobilenet_v2" else "ResNet50 (Trích xuất sâu)"
    )

    st.markdown("---")
    st.markdown("### 📊 Thông tin tập dữ liệu")
    st.markdown("""
    - **Tập dữ liệu**: Kaggle Garbage Classification
    - **Quy mô**: 2.527 ảnh (6 phân lớp)
    - **Độ phân giải**: 224 x 224 RGB
    - **Chia tập**: Train 80% | Val 10% | Test 10%
    """)

    st.markdown("---")
    st.markdown("### 🏷️ 6 Lớp Phân Loại")
    for c, meta in CLASS_METADATA.items():
        st.markdown(f"- {meta['icon']} **{meta['vn_name']}** (`{c}`)")

# Tải mô hình
predictor = get_model(selected_model_type)

# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown('<div class="main-header">♻️ Automated Waste Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hệ thống phân loại rác thải tự động thời gian thực ứng dụng Deep Learning & Transfer Learning</div>', unsafe_allow_html=True)

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Phân Loại Ảnh Tải Lên",
    "📹 Phân Loại Qua Webcam",
    "📈 Báo Cáo & So Sánh Mô Hình",
    "📖 Cẩm Nang Tái Chế Rác Thải"
])

def render_prediction_result(res, input_img):
    col_img, col_info = st.columns([1, 1.3])
    
    with col_img:
        st.image(input_img, caption="Ảnh đầu vào", use_column_width=True)
    
    with col_info:
        # Badge Tái chế / Không tái chế
        if res["is_recyclable"]:
            st.markdown(f'<div class="recyclable-badge">✅ {res["category"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="non-recyclable-badge">⚠️ {res["category"]}</div>', unsafe_allow_html=True)

        st.markdown(f"### {CLASS_METADATA[res['predicted_class']]['icon']} Kết quả: **{res['vn_name']}** (`{res['predicted_class']}`)")
        
        # Thống kê độ tự tin & latency
        m1, m2 = st.columns(2)
        m1.metric("Độ tự tin (Confidence)", res["confidence_percent"])
        m2.metric("Độ trễ suy luận", f"{res['latency_ms']} ms")

        # Hướng dẫn xử lý môi trường
        st.markdown(f"""
        <div class="guide-box">
            <b>📋 Hướng dẫn xử lý:</b> {res['guide']}
        </div>
        """, unsafe_allow_html=True)

        # Biểu đồ Top-K xác suất
        st.markdown("#### 📊 Phân bố xác suất Top 3:")
        df_top = pd.DataFrame(res["top_k"])
        df_top["display_name"] = df_top.apply(lambda r: f"{r['vn_name']} ({r['class']})", axis=1)
        
        fig = px.bar(
            df_top,
            x="confidence",
            y="display_name",
            orientation="h",
            text=df_top["confidence"].apply(lambda v: f"{v*100:.1f}%"),
            color="confidence",
            color_continuous_scale="Blues",
            range_x=[0, 1.05]
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Xác suất dự đoán",
            yaxis_title="",
            margin=dict(l=10, r=10, t=10, b=10),
            height=200,
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 1: UPLOAD ẢNH
# =============================================================================
with tab1:
    st.markdown("#### 📥 Chọn hoặc kéo thả hình ảnh rác thải để nhận diện:")
    uploaded_file = st.file_uploader(
        "Tải lên tệp ảnh (Hỗ trợ JPG, PNG, JPEG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Chọn ảnh chụp rõ nét vật thể rác thải"
    )

    col_btn, _ = st.columns([1, 4])
    sample_btn = col_btn.button("🎲 Thử với ảnh mẫu ngẫu nhiên từ Test set")

    test_dir = BASE_DIR / "data" / "processed" / "test"
    
    selected_img = None
    if uploaded_file is not None:
        selected_img = Image.open(uploaded_file)
    elif sample_btn and test_dir.exists():
        all_test_imgs = list(test_dir.glob("*/*.*"))
        if all_test_imgs:
            random_sample = np.random.choice(all_test_imgs)
            selected_img = Image.open(random_sample)
            st.info(f"Đã chọn mẫu ngẫu nhiên từ nhãn gốc: `{random_sample.parent.name}`")

    if selected_img is not None:
        if predictor is None:
            st.warning("⚠️ Chưa tìm thấy trọng số mô hình đã huấn luyện. Vui lòng hoàn thành quá trình huấn luyện bằng `python train.py`.")
        else:
            with st.spinner("🧠 Đang phân tích đặc trưng hình ảnh..."):
                res = predictor.predict(selected_img)
            render_prediction_result(res, selected_img)

# =============================================================================
# TAB 2: WEBCAM REAL-TIME
# =============================================================================
with tab2:
    st.markdown("#### 📷 Nhận diện trực tiếp qua Camera thiết bị:")
    st.write("Bật camera, đưa vật phẩm rác thải trước ống kính và chụp ảnh để phân loại tức thì:")
    
    camera_photo = st.camera_input("Chụp ảnh từ webcam")
    
    if camera_photo is not None:
        cam_img = Image.open(camera_photo)
        if predictor is None:
            st.warning("⚠️ Chưa tìm thấy trọng số mô hình đã huấn luyện.")
        else:
            with st.spinner("🔍 Đang nhận diện vật thể..."):
                res = predictor.predict(cam_img)
            render_prediction_result(res, cam_img)

# =============================================================================
# TAB 3: BÁO CÁO & SO SÁNH HIỆU NĂNG MÔ HÌNH
# =============================================================================
with tab3:
    st.markdown("### 📊 Đánh Giá & So Sánh Hiệu Năng Mô Hình Học Sâu")
    
    mob_metrics_path = OUTPUT_DIR / "mobilenet_v2_metrics.json"
    res_metrics_path = OUTPUT_DIR / "resnet50_metrics.json"
    
    col_stat1, col_stat2 = st.columns(2)

    with col_stat1:
        st.markdown("#### 📱 MobileNetV2 (Tối ưu Real-Time)")
        if mob_metrics_path.exists():
            with open(mob_metrics_path, "r", encoding="utf-8") as f:
                mob_data = json.load(f)
            st.metric("Test Accuracy", f"{mob_data['accuracy']*100:.2f}%")
            st.metric("Macro F1-Score", f"{mob_data['macro_avg']['f1_score']*100:.2f}%")
            st.metric("Weighted F1-Score", f"{mob_data['weighted_avg']['f1_score']*100:.2f}%")
        else:
            st.info("Chạy `python evaluate.py --model mobilenet_v2` để cập nhật số liệu.")

        mob_cm_path = OUTPUT_DIR / "mobilenet_v2_confusion_matrix.png"
        if mob_cm_path.exists():
            st.image(str(mob_cm_path), caption="Confusion Matrix - MobileNetV2", use_column_width=True)

        mob_hist_path = OUTPUT_DIR / "mobilenet_v2_training_history.png"
        if mob_hist_path.exists():
            st.image(str(mob_hist_path), caption="Learning Curve - MobileNetV2", use_column_width=True)

    with col_stat2:
        st.markdown("#### 🏢 ResNet50 (Trích Xuất Sâu)")
        if res_metrics_path.exists():
            with open(res_metrics_path, "r", encoding="utf-8") as f:
                res_data = json.load(f)
            st.metric("Test Accuracy", f"{res_data['accuracy']*100:.2f}%")
            st.metric("Macro F1-Score", f"{res_data['macro_avg']['f1_score']*100:.2f}%")
            st.metric("Weighted F1-Score", f"{res_data['weighted_avg']['f1_score']*100:.2f}%")
        else:
            st.info("Chạy `python evaluate.py --model resnet50` để cập nhật số liệu.")

        res_cm_path = OUTPUT_DIR / "resnet50_confusion_matrix.png"
        if res_cm_path.exists():
            st.image(str(res_cm_path), caption="Confusion Matrix - ResNet50", use_column_width=True)

        res_hist_path = OUTPUT_DIR / "resnet50_training_history.png"
        if res_hist_path.exists():
            st.image(str(res_hist_path), caption="Learning Curve - ResNet50", use_column_width=True)

# =============================================================================
# TAB 4: CẨM NANG PHÂN LOẠI
# =============================================================================
with tab4:
    st.markdown("### 📚 Cẩm Nang Phân Loại & Hướng Dẫn Tái Chế Rác Thải")
    st.write("Tra cứu quy chuẩn xử lý môi trường cho 6 loại rác thải phổ biến:")

    cols = st.columns(3)
    for idx, (cls, meta) in enumerate(CLASS_METADATA.items()):
        col = cols[idx % 3]
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {meta['color']}; min-height: 200px;">
                <h4 style="margin-top:0;">{meta['icon']} {meta['vn_name']} (<code>{cls}</code>)</h4>
                <p><b>Nhóm:</b> {'♻️ Tái chế được' if meta['is_recyclable'] else '🗑️ Rác thông thường'}</p>
                <p style="font-size:0.9rem; color:#475569;"><b>Quy trình xử lý:</b> {meta['guide']}</p>
            </div>
            """, unsafe_allow_html=True)
