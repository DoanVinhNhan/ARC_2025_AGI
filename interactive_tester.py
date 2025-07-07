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

def create_grid_html(grid):
    """Tạo mã HTML để biểu diễn grid bằng CSS Grid Layout."""
    grid = np.array(grid)
    rows, cols = grid.shape
    if rows == 0 or cols == 0: return ""
    
    html = (
        f"<div style='"
        f"display: grid; "
        f"grid-template-columns: repeat({cols}, 1fr); "
        f"grid-template-rows: repeat({rows}, 1fr); "
        f"width: 100%; "
        f"aspect-ratio: {cols} / {rows}; "
        f"border: 1px solid grey; "
        f"gap: 1px; "
        f"background-color: grey;'"
        ">"
    )
    for r in range(rows):
        for c in range(cols):
            color = ARC_CSS_COLORS.get(grid[r, c], 'white')
            html += f"<div style='background-color: {color};'></div>"
    html += "</div>"
    return html

# --- HÀM MỚI: TẠO HTML CHO CẢ MỘT CẶP INPUT/OUTPUT ---
def create_pair_html_view(input_grid_data, output_grid_data, title):
    """Tạo một khối HTML hoàn chỉnh cho một cặp, sử dụng Flexbox để căn giữa."""
    
    # Tạo HTML cho grid input và thông tin của nó
    input_grid_html = create_grid_html(input_grid_data)
    input_rows, input_cols = np.array(input_grid_data).shape
    input_html_block = (
        f"<div>"
        f"<b>{title}: Input</b>"
        f"{input_grid_html}"
        f"<div style='font-size: 12px; color: grey;'>Size: {input_rows}x{input_cols}</div>"
        f"</div>"
    )

    # Tạo HTML cho grid output và thông tin của nó
    if output_grid_data is not None:
        output_grid_html = create_grid_html(output_grid_data)
        output_rows, output_cols = np.array(output_grid_data).shape
        output_html_block = (
            f"<div>"
            f"<b>Output</b>"
            f"{output_grid_html}"
            f"<div style='font-size: 12px; color: grey;'>Size: {output_rows}x{output_cols}</div>"
            f"</div>"
        )
    else: # Trường hợp không có output (dành cho test prediction)
        output_html_block = "<div style='font-size: 50px; text-align: center;'>?</div>"

    # Bọc tất cả trong một Flexbox container để căn giữa theo chiều dọc
    full_pair_html = (
        f"<div style='display: flex; align-items: center; justify-content: space-between; width: 100%;'>"
        f"<div style='flex: 1;'>{input_html_block}</div>"
        f"<div style='flex: 0.2; text-align: center; font-size: 24px;'>&rarr;</div>"
        f"<div style='flex: 1;'>{output_html_block}</div>"
        f"</div>"
    )
    
    return full_pair_html


@st.cache_data
def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def dummy_solver(task):
    predictions = []
    for test_pair in task['test']:
        test_input_grid = np.array(test_pair['input'])
        predicted_grid = np.fliplr(test_input_grid)
        predictions.append(predicted_grid.tolist())
    return predictions


# === Giao diện ứng dụng Streamlit ===
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("ARC - Công cụ Thử nghiệm Hệ thống 🚀")

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
    
    col_train, col_test = st.columns(2, gap="large")

    # === CỘT BÊN TRÁI: HIỂN THỊ TRAIN ===
    with col_train:
        st.header("Train")
        task_data = tasks[task_id_input]
        if not task_data['train']:
            st.warning("Task này không có ví dụ train.")
        for i, pair in enumerate(task_data['train']):
            # Gọi hàm mới để tạo HTML cho cả cặp
            pair_html = create_pair_html_view(pair['input'], pair['output'], f'Train {i}')
            st.markdown(pair_html, unsafe_allow_html=True)
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
                prediction_grid = None
                if f'prediction_{task_id_input}' in st.session_state:
                    prediction_grid = st.session_state[f'prediction_{task_id_input}'][i]
                
                # Gọi hàm mới để tạo HTML cho cặp test
                pair_html = create_pair_html_view(pair['input'], prediction_grid, f'Test {i}')
                st.markdown(pair_html, unsafe_allow_html=True)
                st.markdown("---")