import streamlit as st
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# ==========================================
# 页面基础设置 (必须在第一行)
# ==========================================
st.set_page_config(
    page_title="RF & Q-RF Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 模型缓存加载模块 (关键：避免每次交互重复加载模型)
# ==========================================
@st.cache_resource
def load_dtnn_models():
    graph_1pct = tf.Graph()
    graph_2pct = tf.Graph()
    model_1pct = None
    model_2pct = None

    try:
        from DTNN1 import DTNN
        if os.path.exists("./rf_Q_model_1%"):
            with graph_1pct.as_default():
                model_1pct = DTNN(model_path="./rf_Q_model_1%")

        if os.path.exists("./rf_Q_model_2%"):
            with graph_2pct.as_default():
                model_2pct = DTNN(model_path="./rf_Q_model_2%")

        return graph_1pct, model_1pct, graph_2pct, model_2pct
    except Exception as e:
        return None, None, None, str(e)


# ==========================================
# 辅助函数：支护类别判定
# ==========================================
def determine_support_category(q_val, span):
    x = float(q_val)
    y = float(span)

    if (y >= 4 and x <= 100 and y < (11 * x / 96 + 85 / 24)):
        return "Category ①", "Unsupported or spot bolting"
    if (y <= 30 and x <= 100 and y >= (11 * x / 96 + 85 / 24) and y < (0.44 * x - 5.2)):
        return "Category ②", "Spot bolting"
    if (y <= 30 and y >= 4 and y >= (0.44 * x - 5.2) and y >= (11 * x / 96 + 85 / 24) and y < (13 * x / 7 + 15 / 7)):
        return "Category ③", "Systematic bolting + Fibre reinforced sprayed concrete (5-6 cm)"
    if (y <= 30 and y >= 4 and y >= (13 * x / 7 + 15 / 7) and y < (104 * x / 23 + 66 / 23)):
        return "Category ④", "Fibre reinforced sprayed concrete (6-9 cm) + Bolting"
    if (y <= 30 and y >= 4 and y >= (104 * x / 23 + 66 / 23) and y < (2600 * x / 143 + 390 / 143)):
        return "Category ⑤", "Fibre reinforced sprayed concrete (9-12 cm) + Bolting"
    if (y <= 30 and y >= 4 and y >= (2600 * x / 143 + 390 / 143) and y < (13000 * x / 241 + 730 / 241)):
        return "Category ⑥", "Fibre reinforced sprayed concrete (12-15 cm) + Reinforced ribs of sprayed concrete and bolting (RRS-A) + Bolting"
    if (y <= 30 and y >= 4 and y >= (13000 * x / 241 + 730 / 241) and y < (26000 * x / 179 + 690 / 179)):
        return "Category ⑦", "Fibre reinforced sprayed concrete (>15 cm) + Reinforced ribs of sprayed concrete and bolting (RRS-B) + Bolting"
    if (y <= 30 and y >= 4 and y >= (26000 * x / 179 + 690 / 179) and y < (5000 * x / 33 + 490 / 33)):
        return "Category ⑧", "Fibre reinforced sprayed concrete (>25 cm) + Double layer ribs of sprayed concrete and bolting (RRS-C) + Bolting"
    if (y <= 30 and x >= 0.001 and y >= (5000 * x / 33 + 490 / 33)):
        return "Category ⑨", "Special evaluation required"

    return "Unknown Category", "Parameters outside defined support regions"


# 查找图片的辅助函数
def find_image(base_name, exts=[".jpg", ".png", ".jpeg", ".PNG"]):
    for ext in exts:
        full_path = base_name + ext
        if os.path.exists(full_path):
            return full_path
    return None


# ==========================================
# 页面布局开始
# ==========================================
graph_1pct, model_1pct, graph_2pct, model_2pct = load_dtnn_models()

# 检查模型是否加载成功
if isinstance(model_2pct, str):  # 意味着捕获到了错误
    st.error(f"⚠️ 无法加载 DTNN1.py 或模型文件。错误信息: {model_2pct}")
    st.stop()

# 定义左右两列，比例大约为 4.5 : 5.5
col_left, col_right = st.columns([4.5, 5.5], gap="large")

# ----------------- 左侧：参考图 -----------------
with col_left:
    img_q = find_image("Q value")
    if img_q:
        st.image(img_q, use_container_width=True, caption="Q-Value Chart")
    else:
        st.info("Q Value chart image not found.")

    st.divider()  # 分割线

    img_support = find_image("Support Method")
    if img_support:
        st.image(img_support, use_container_width=True, caption="Support Method Chart")
    else:
        st.info("Support Method image not found.")

# ----------------- 右侧：控制与结果 -----------------
with col_right:
    # 顶部 Logos
    logo_col1, logo_col2 = st.columns(2)
    img_monash = find_image("Monash", [".png", ".PNG", ".jpg"])
    img_seu = find_image("SEU", [".png", ".PNG", ".jpg"])

    with logo_col1:
        if img_monash: st.image(img_monash, width=200)
    with logo_col2:
        if img_seu: st.image(img_seu, width=200)

    st.markdown("---")

    # Step 1: Model Select
    st.subheader("Step 1: Select Model")
    selected_model = st.selectbox(
        "Choose Model Configuration:",
        ["rf_Q_model_2%", "rf_Q_model_1%"],
        index=0
    )

    if model_1pct and model_2pct:
        st.success("✅ Models loaded successfully and ready.")
    else:
        st.warning("⚠️ Some models might be missing in the directory.")

    st.markdown("---")

    # Step 2: Input Parameters
    st.subheader("Step 2: Input Parameters (Integer Only)")

    # 按照原来的滑块设置
    val_depth = st.slider("Depth (Meters)", min_value=40, max_value=240, value=100, step=1)
    val_mag = st.slider("Magnitude", min_value=1, max_value=8, value=6, step=1)
    val_q = st.slider("Q-value", min_value=4, max_value=100, value=50, step=1)
    val_span = st.slider("Span (Meters)", min_value=4, max_value=30, value=15, step=1)

    # 运行预测的按钮
    run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

    st.markdown("---")
    st.subheader("Step 3: Results")

    # 如果按下了按钮，或者页面重新加载
    if run_btn:
        if not model_1pct or not model_2pct:
            st.error("Models are not fully loaded. Cannot run prediction.")
        else:
            try:
                # 准备输入数据
                q_input_val = np.log10(val_q)
                input_data = np.array([[float(val_depth), float(val_mag), float(val_span), float(q_input_val), 0.0]])

                # === 1. 运行两个模型 ===
                with graph_1pct.as_default():
                    pred_1 = model_1pct.nn_predict(input_data, 1, 1)
                    rf_1 = pred_1[0][0]

                with graph_2pct.as_default():
                    pred_2 = model_2pct.nn_predict(input_data, 1, 1)
                    rf_2 = pred_2[0][0]

                # === 2. 获取当前选择的模型，执行逻辑 ===
                raw_rf = 0.0
                log_msg = ""

                if selected_model == "rf_Q_model_2%":
                    raw_rf = rf_2
                    log_msg = "Model: 2%"
                elif selected_model == "rf_Q_model_1%":
                    log_msg = f"Model 1%: {rf_1:.4f} | Model 2%: {rf_2:.4f}"
                    if rf_1 > rf_2:
                        raw_rf = rf_2
                        log_msg += " -> Using 2% (Logic: 1% > 2%)"
                    else:
                        raw_rf = rf_1
                        log_msg += " -> Using 1%"

                # === 3. 约束逻辑 ===
                msg = log_msg
                if val_depth > 300:
                    val_rf = 1.0
                    msg += " | Depth > 300 -> RF=1.0"
                elif raw_rf > 1.0:
                    val_rf = 1.0
                    msg += " | RF Capped at 1.0"
                elif raw_rf < 0.0:
                    val_rf = 0.0
                    msg += " | RF clipped to 0.0"
                else:
                    val_rf = raw_rf

                # === 4. 计算其余结果 ===
                if val_rf == 0.0:
                    val_q_rf_str = "NaN"
                    cat_title = "Category ⑨"
                    cat_desc = "Special evaluation required"
                    anchor_s_str = "NaN"
                    anchor_l_str = "NaN"
                    msg += " | RF=0 -> Invalid Q"
                else:
                    val_q_rf = val_q * val_rf
                    val_q_rf_str = f"{val_q_rf:.4f}"
                    cat_title, cat_desc = determine_support_category(val_q_rf, val_span)

                    try:
                        log_q = np.log10(val_q_rf)
                        val_s = 10 ** (0.24 + 0.12 * log_q)
                        anchor_s_str = f"{val_s:.2f} m"
                    except:
                        anchor_s_str = "Err"

                    val_l = 1.4 + (0.184 * val_span)
                    anchor_l_str = f"{val_l:.2f} m"

                val_rf_str = f"{val_rf:.4f}"

                # ==========================================
                # 自定义 CSS 网格布局渲染结果 (还原 Tkinter 颜色)
                # ==========================================

                # 第一行
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown(f"""
                    <div style="border: 1px solid #ccc; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                        <div style="background-color: #e1e1e1; font-weight: bold; padding: 5px; border-radius: 4px 4px 0 0;">RF (Neural Net)</div>
                        <h2 style="color: #0055aa; margin: 15px 0;">{val_rf_str}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with res_col2:
                    st.markdown(f"""
                    <div style="border: 1px solid #ccc; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                        <div style="background-color: #e1e1e1; font-weight: bold; padding: 5px; border-radius: 4px 4px 0 0;">Calculated Q (Q × RF)</div>
                        <h2 style="color: #228b22; margin: 15px 0;">{val_q_rf_str}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                # 第二行
                res_col3, res_col4 = st.columns(2)
                with res_col3:
                    st.markdown(f"""
                    <div style="border: 1px solid #ccc; border-radius: 5px; text-align: center; height: 130px;">
                        <div style="background-color: #ffebcd; font-weight: bold; padding: 5px; border-radius: 4px 4px 0 0;">Recommended Support Category</div>
                        <h3 style="color: #d2691e; margin-top: 10px; margin-bottom: 5px;">{cat_title}</h3>
                        <p style="font-size: 14px; color: #555; padding: 0 10px;">{cat_desc}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with res_col4:
                    st.markdown(f"""
                    <div style="border: 1px solid #ccc; border-radius: 5px; text-align: center; height: 130px;">
                        <div style="background-color: #e0ffff; font-weight: bold; padding: 5px; border-radius: 4px 4px 0 0;">Anchor Length & Spacing</div>
                        <div style="margin-top: 15px;">
                            <p style="font-size: 18px; margin: 5px 0;"><b>Length (L):</b> <span style="color: black; font-weight: bold;">{anchor_l_str}</span></p>
                            <p style="font-size: 18px; margin: 5px 0;"><b>Spacing (s):</b> <span style="color: black; font-weight: bold;">{anchor_s_str}</span></p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # 打印日志
                st.caption(f"📝 **Log:** {msg}")

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
    else:
        st.info("👆 请调整上方参数，并点击 'Run Prediction' 查看结果。")