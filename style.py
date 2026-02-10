import os
import base64
import streamlit as st


def get_base64_font(font_path):
    if os.path.exists(font_path):
        with open(font_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def apply_custom_styles():
    icon_css_path = "icons/bootstrap-icons.css"
    font_path_woff2 = "icons/fonts/bootstrap-icons.woff2"
    vazir_font_path = "icons/fonts/Vazirmatn.woff2"
    font_base64 = get_base64_font(font_path_woff2)
    vazir_base64 = get_base64_font(vazir_font_path)
    css_content = ""
    if os.path.exists(icon_css_path):
        with open(icon_css_path, "r") as f:
            css_content = f.read()
    st.markdown(
        f"""
        <style>
        @font-face {{ font-family: 'bootstrap-icons'; src: url(data:font/woff2;base64,{font_base64}) format('woff2'); }}
        @font-face {{ font-family: 'Vazirmatn'; src: url(data:font/woff2;base64,{vazir_base64}) format('woff2'); }}
        {css_content}

        /* تنظیمات پایه و جلوگیری از اسکرول افقی /
        html, body, [data-testid="stAppViewContainer"] {{
            font-family: 'Vazirmatn', sans-serif !important;
            direction: rtl !important;
            text-align: right !important;
            overflow-x: hidden !important; / جلوگیری از اسکرول افقی کل صفحه /
        }}

        / اصلاح نمایش متن چت و شکستن خطوط طولانی /
        .stChatMessage div, .stChatMessage p {{
            direction: rtl !important;
            text-align: right !important;
            white-space: pre-wrap !important; / حفظ فاصله‌ها و شکستن خط /
            word-wrap: break-word !important; / شکستن کلمات خیلی طولانی */
            overflow-wrap: break-word !important;
        }}

        [data-testid="stSidebar"][aria-expanded="false"] > div {{ display: none !important; }}
        [data-testid="stSidebar"] {{ direction: rtl !important; background-color: #111827 !important; }}

        [data-testid="stSidebarCollapseControl"] button svg,
        [data-testid="stSidebarCollapsedControl"] button svg {{ transform: scaleX(-1) !important; }}

        [data-testid="stSidebarCollapsedControl"] {{ right: 10px !important; left: auto !important; background-color: #111827 !important; }}

        .monitor-card {{ background: #064e3b; color: #34d399; padding: 12px; border-radius: 10px; margin-bottom: 8px; border-right: 5px solid #10b981; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
