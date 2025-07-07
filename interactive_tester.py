# interactive_tester.py

import streamlit as st
import json
import numpy as np

# Bảng màu CSS (giữ nguyên)
ARC_CSS_COLORS = {
    0: 'rgb(0, 0, 0)', 1: 'rgb(31, 147, 255)', 2: 'rgb(248, 59, 48)',
    3: 'rgb(79, 204, 48)', 4: 'rgb(255, 220, 0)', 5: 'rgb(153, 153, 153)',
    6: 'rgb(229, 58, 163)', 7: 'rgb(255, 133, 27)', 8: 'rgb(136, 216, 241)',
    9: 'rgb(146, 18, 49)'
}

def create_grid_html(grid, cell_size=20):
    """Tạo một chuỗi HTML <table> để biểu diễn grid."""
    grid = np.array(grid)
    html = "<table style='border-collapse: collapse; border: 1px solid grey;'>"
    for row in grid:
        html += "<tr>"
        for cell_color_index in row:
            color = ARC_CSS_COLORS.get(cell_color_index, 'white')
            html += (
                f"<td style='width:{cell_size}px; height:{cell_size}px; "
                f"background-color:{color}; border: 1px solid grey;'>"
                "</td>"
            )
        html += "</tr>"
    html += "</table>"
    return html

@st.cache_data
def load_data(file_path):
    """Tải và cache dữ liệu từ file JSON."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def dummy_solver(task):
    """Hàm giải quyết giả lập."""
    predictions = []
    for test_pair in task['test']:
        test_input_grid = np.array(test_pair['input'])
        predicted_grid = np.fliplr(test_input_grid)
        predictions.append(predicted_grid.tolist())
    return predictions

def display_grid_with_info(grid_data, title):
    """Hàm tiện ích để hiển thị một grid cùng với tiêu đề và kích thước."""
    st.markdown(f"**{title}**")
    
    # Bọc bảng HTML trong một div có thanh cuộn ngang tự động
    grid_html = create_grid_html(grid_data)
    st.markdown(f"<div style='overflow-x: auto; width: 100%;'>{grid_html}</div>", unsafe_allow_html=True)
    
    # Hiển thị kích thước
    rows, cols = np.array(grid_data).shape
    st.caption(f"Size: {rows}x{cols}")


# === Bắt đầu giao diện ứng dụng ===
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("ARC - Công cụ Thử nghiệm Hệ thống")

tasks = load_data('data/arc-agi_training_challenges.json')

if tasks is None:
    st.error("Lỗi: Không tìm thấy file dữ liệu tại 'data/arc-agi_training_challenges.json'.")
else:
    # --- Thanh bên (Sidebar) ---
    st.sidebar.header("Điều khiển")
    task_ids = list(tasks.keys())
    task_id_input = st.sidebar.selectbox(
        "Chọn hoặc nhập một Task ID:",
        options=task_ids,
        index=task_ids.index('007bbfb7')
    )

    col_train, col_test = st.columns(2)

    # === CỘT BÊN TRÁI: HIỂN THỊ TRAIN ===
    with col_train:
        st.header("Train")
        task_data = tasks[task_id_input]
        if not task_data['train']:
            st.warning("Task này không có ví dụ train.")
        for i, pair in enumerate(task_data['train']):
            st.write(f"**Cặp ví dụ {i}**")
            sub_col_1, sub_col_2, sub_col_3 = st.columns([1, 0.2, 1])
            with sub_col_1:
                display_grid_with_info(pair['input'], "Input")
            with sub_col_2:
                # Dùng HTML/CSS để căn giữa mũi tên theo chiều dọc
                st.markdown("<div style='display: flex; align-items: center; justify-content: center; height: 100%; font-size: 24px;'>&rarr;</div>", unsafe_allow_html=True)
            with sub_col_3:
                display_grid_with_info(pair['output'], "Output")
            st.markdown("---")

    # === CỘT BÊN PHẢI: HIỂN THỊ TEST ===
    with col_test:
        st.header("Test")
        task_data = tasks[task_id_input]
        
        if st.button("Chạy Hệ thống để Dự đoán"):
            predictions = dummy_solver(task_data)
            st.session_state[f'prediction_{task_id_input}'] = predictions

        st.markdown("---")

        if not task_data['test']:
            st.warning("Task này không có bài toán test.")
        else:
            for i, pair in enumerate(task_data['test']):
                st.write(f"**Bài toán {i}**")
                sub_col_1, sub_col_2, sub_col_3 = st.columns([1, 0.2, 1])
                
                with sub_col_1:
                    display_grid_with_info(pair['input'], "Input")

                with sub_col_2:
                    st.markdown("<div style='display: flex; align-items: center; justify-content: center; height: 100%; font-size: 24px;'>&rarr;</div>", unsafe_allow_html=True)
                
                with sub_col_3:
                    st.markdown("**Prediction**")
                    if f'prediction_{task_id_input}' in st.session_state:
                        predicted_grid = st.session_state[f'prediction_{task_id_input}'][i]
                        # Hiển thị grid dự đoán và thông tin của nó
                        grid_html = create_grid_html(predicted_grid)
                        st.markdown(f"<div style='overflow-x: auto; width: 100%;'>{grid_html}</div>", unsafe_allow_html=True)
                        rows, cols = np.array(predicted_grid).shape
                        st.caption(f"Size: {rows}x{cols}")
                    else:
                        st.markdown("<div style='font-size: 50px;'>?</div>", unsafe_allow_html=True)
                
                st.markdown("---")