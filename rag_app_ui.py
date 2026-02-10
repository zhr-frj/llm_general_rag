# # import streamlit as st
# # import torch, gc, json, hashlib, time, uuid, faiss, psutil, os, base64, re
# # from pathlib import Path
# # from pdf2image import convert_from_bytes
# # import pytesseract
# # from concurrent.futures import ThreadPoolExecutor
# # from vector_manager import load_vectorstore_on_gpu
# # from setup_models import setup_llm_and_embeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_core.documents import Document

# # # --- تنظیمات اولیه صفحه ---
# # st.set_page_config(page_title="دستیار هوشمند سازمانی", layout="wide")


# # def get_base64_font(font_path):
# #     if os.path.exists(font_path):
# #         with open(font_path, "rb") as f:
# #             return base64.b64encode(f.read()).decode()
# #     return ""


# # def apply_custom_styles():
# #     icon_css_path = "icons/bootstrap-icons.css"
# #     font_path_woff2 = "icons/fonts/bootstrap-icons.woff2"
# #     vazir_font_path = "icons/fonts/Vazirmatn.woff2"
# #     font_base64 = get_base64_font(font_path_woff2)
# #     vazir_base64 = get_base64_font(vazir_font_path)
# #     css_content = ""
# #     if os.path.exists(icon_css_path):
# #         with open(icon_css_path, "r") as f:
# #             css_content = f.read()

# #     st.markdown(
# #         f"""
# #     <style>
# #     @font-face {{ font-family: 'bootstrap-icons'; src: url(data:font/woff2;base64,{font_base64}) format('woff2'); }}
# #     @font-face {{ font-family: 'Vazirmatn'; src: url(data:font/woff2;base64,{vazir_base64}) format('woff2'); }}
# #     {css_content}

# #     /* تنظیم کلی جهت صفحه و فونت */
# #     html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
# #         font-family: 'Vazirmatn', sans-serif !important;
# #         direction: rtl !important;
# #         text-align: right !important;
# #     }}

# #     /* اصلاح رفتار سایدبار در حالت RTL - حذف پوزیشن فیکس دستی برای جلوگیری از باگ بصری */
# #     [data-testid="stSidebar"] {{
# #         background-color: #111827 !important;
# #         min-width: 320px !important;
# #         max-width: 320px !important;
# #         direction: rtl !important;
# #     }}

# #     /* فیکس کردن دکمه فلش بازکننده سایدبار در سمت راست */
# #     [data-testid="stSidebarCollapsedControl"] {{
# #         right: 0 !important;
# #         left: auto !important;
# #         background-color: #111827 !important;
# #         display: flex !important;
# #         justify-content: center !important;
# #         border-radius: 5px 0 0 5px !important;
# #     }}

# #     /* اطمینان از اینکه محتوای اصلی کل فضا را می‌گیرد و زیر سایدبار نمی‌رود */
# #     [data-testid="stMainViewContainer"] {{
# #         width: 100% !important;
# #     }}

# #     .stChatMessage {{ direction: rtl !important; text-align: right !important; }}
# #     .monitor-card {{ background: #064e3b; color: #34d399; padding: 12px; border-radius: 10px; margin-bottom: 8px; border-right: 5px solid #10b981; }}
# #     </style>
# #     """,
# #         unsafe_allow_html=True,
# #     )


# # def clean_ocr_text(text):
# #     text = re.sub(r"[°τµ~|—_]{2,}", "", text)
# #     text = re.sub(r"\s+", " ", text)
# #     return text.strip()


# # apply_custom_styles()

# # DATA_DIR = Path("data")
# # INDEX_PATH = DATA_DIR / "vectorstore"
# # METADATA_DIR = DATA_DIR / "metadata"
# # for p in [METADATA_DIR, INDEX_PATH]:
# #     p.mkdir(parents=True, exist_ok=True)

# # with st.spinner("⏳ در حال بیدار کردن مدل‌ها..."):
# #     embeddings, llm_engine, prompt_template, (rerank_m, rerank_t) = (
# #         setup_llm_and_embeddings()
# #     )

# # if "vectorstore" not in st.session_state:
# #     st.session_state.vectorstore = load_vectorstore_on_gpu(
# #         str(INDEX_PATH), embeddings, 1
# #     )

# # if "full_texts" not in st.session_state:
# #     st.session_state.full_texts = {}
# #     for meta_file in METADATA_DIR.glob("*.json"):
# #         with open(meta_file, "r") as f:
# #             data = json.load(f)
# #             if "full_content" in data:
# #                 st.session_state.full_texts[data["name"]] = data["full_content"]

# # # --- مدیریت سایدبار ---
# # with st.sidebar:
# #     # استفاده از تگ h3 ساده برای جلوگیری از تداخل نمایش آیکون
# #     st.markdown(
# #         '<h3 style="color: white; direction: rtl; text-align: right;"><i class="bi bi-cpu-fill"></i> پنل مدیریت</h3>',
# #         unsafe_allow_html=True,
# #     )

# #     with st.expander("📊 مانیتورینگ سخت‌افزار", expanded=False):
# #         for i in range(torch.cuda.device_count()):
# #             used = torch.cuda.memory_reserved(i) / 1024**3
# #             st.markdown(
# #                 f'<div class="monitor-card"><i class="bi bi-gpu-card"></i> <b>کارت گرافیک {i}</b><br>مصرف: {used:.1f} GB</div>',
# #                 unsafe_allow_html=True,
# #             )

# #         st.markdown(
# #             f'<div class="monitor-card" style="background:#1e293b;"><i class="bi bi-cpu"></i> <b>رم سیستم: {psutil.virtual_memory().percent}%</b></div>',
# #             unsafe_allow_html=True,
# #         )

# #     st.divider()
# #     files = st.file_uploader("آپلود PDF", type="pdf", accept_multiple_files=True)

# #     if files and st.button("🪄 شروع پردازش"):
# #         new_docs = []
# #         with st.status("🚀 در حال بررسی و پردازش هوشمند...", expanded=True) as status:
# #             for f in files:
# #                 f_bytes = f.read()
# #                 f_hash = hashlib.md5(f_bytes).hexdigest()
# #                 meta_path = METADATA_DIR / f"{f_hash}.json"

# #                 if meta_path.exists():
# #                     st.toast(f"✅ فایل {f.name} قبلاً در دیتابیس موجود بود.", icon="💾")
# #                     with open(meta_path, "r") as m:
# #                         combined_text = json.load(m)["full_content"]
# #                 else:
# #                     status.write(f"🔍 در حال استخراج متن جدید: {f.name}")
# #                     imgs = convert_from_bytes(f_bytes, dpi=150)
# #                     with ThreadPoolExecutor(max_workers=4) as exe:
# #                         texts = list(
# #                             exe.map(
# #                                 lambda img: pytesseract.image_to_string(
# #                                     img, lang="fas+eng"
# #                                 ),
# #                                 imgs,
# #                             )
# #                         )

# #                     combined_text = clean_ocr_text("\n\n".join(texts))
# #                     with open(meta_path, "w") as m:
# #                         json.dump({"name": f.name, "full_content": combined_text}, m)
# #                     st.toast(f"✨ فایل {f.name} با موفقیت پردازش شد.", icon="✅")

# #                 st.session_state.full_texts[f.name] = combined_text
# #                 new_docs.append(
# #                     Document(page_content=combined_text, metadata={"source": f.name})
# #                 )

# #             if new_docs:
# #                 splits = RecursiveCharacterTextSplitter(
# #                     chunk_size=800, chunk_overlap=200
# #                 ).split_documents(new_docs)
# #                 vs = FAISS.from_documents(splits, embeddings)
# #                 vs.index = faiss.index_gpu_to_cpu(vs.index)
# #                 vs.save_local(str(INDEX_PATH))
# #                 st.session_state.vectorstore = load_vectorstore_on_gpu(
# #                     str(INDEX_PATH), embeddings, 1
# #                 )
# #                 status.update(label="✅ دیتابیس بروزرسانی شد!", state="complete")
# #                 st.rerun()

# # # --- بخش اصلی گفتگو ---
# # st.title("🏢 دستیار هوشمند مدیریت دانش")
# # if "messages" not in st.session_state:
# #     st.session_state.messages = []

# # for m in st.session_state.messages:
# #     with st.chat_message(m["role"], avatar="👤" if m["role"] == "user" else "🤖"):
# #         st.markdown(
# #             f'<div style="text-align: right; direction: rtl;">{m["content"]}</div>',
# #             unsafe_allow_html=True,
# #         )

# # if prompt := st.chat_input("سوال خود را بپرسید..."):
# #     st.session_state.messages.append({"role": "user", "content": prompt})
# #     with st.chat_message("user", avatar="👤"):
# #         st.markdown(
# #             f'<div style="text-align: right; direction: rtl;">{prompt}</div>',
# #             unsafe_allow_html=True,
# #         )

# #     with st.chat_message("assistant", avatar="🤖"):
# #         if st.session_state.vectorstore:
# #             with st.spinner("🔍 در حال تحلیل..."):
# #                 docs = st.session_state.vectorstore.similarity_search(prompt, k=15)
# #                 pairs = [[prompt, d.page_content] for d in docs]
# #                 inputs = rerank_t(
# #                     pairs,
# #                     padding=True,
# #                     truncation=True,
# #                     return_tensors="pt",
# #                     max_length=512,
# #                 ).to("cuda:1")

# #                 with torch.no_grad():
# #                     scores = rerank_m(**inputs).logits.view(-1).float()

# #                 context = "\n\n".join(
# #                     [
# #                         docs[i].page_content
# #                         for i in torch.argsort(scores, descending=True)[:8]
# #                     ]
# #                 )
# #                 chain = prompt_template | llm_engine
# #                 response = chain.invoke({"context": context, "question": prompt})

# #                 # تمیزکاری خروجی
# #                 ans = (
# #                     response.split("پاسخ جامع و تشریحی:")[-1].strip()
# #                     if "پاسخ جامع و تشریحی:" in response
# #                     else response
# #                 )
# #                 ans = re.sub(r"\*+", "", ans).strip()

# #                 st.markdown(
# #                     f'<div style="text-align: right; direction: rtl;">{ans}</div>',
# #                     unsafe_allow_html=True,
# #                 )
# #                 st.session_state.messages.append({"role": "assistant", "content": ans})

# #     torch.cuda.empty_cache()
# #     gc.collect()


# ##rag_app_ui.py


# import streamlit as st
# import torch, gc, json, hashlib, time, uuid, faiss, psutil, os, base64, re
# from pathlib import Path
# from pdf2image import convert_from_bytes
# import pytesseract
# from concurrent.futures import ThreadPoolExecutor
# from vector_manager import load_vectorstore_on_gpu
# from setup_models import setup_llm_and_embeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

# # --- وارد کردن فایل استایل جداگانه ---
# from style import apply_custom_styles

# # --- تنظیمات اولیه صفحه ---
# st.set_page_config(page_title="دستیار هوشمند سازمانی", layout="wide")

# # --- اعمال استایل‌ها ---
# apply_custom_styles()


# def clean_ocr_text(text):
#     text = re.sub(r"[°τµ~|—_]{2,}", "", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# DATA_DIR = Path("data")
# INDEX_PATH = DATA_DIR / "vectorstore"
# METADATA_DIR = DATA_DIR / "metadata"
# for p in [METADATA_DIR, INDEX_PATH]:
#     p.mkdir(parents=True, exist_ok=True)

# with st.spinner("⏳ در حال بیدار کردن مدل‌ها..."):
#     embeddings, llm_engine, prompt_template, (rerank_m, rerank_t) = (
#         setup_llm_and_embeddings()
#     )

# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = load_vectorstore_on_gpu(
#         str(INDEX_PATH), embeddings, 1
#     )

# if "full_texts" not in st.session_state:
#     st.session_state.full_texts = {}
#     for meta_file in METADATA_DIR.glob("*.json"):
#         with open(meta_file, "r") as f:
#             data = json.load(f)
#             if "full_content" in data:
#                 st.session_state.full_texts[data["name"]] = data["full_content"]

# # --- مدیریت سایدبار ---
# with st.sidebar:
#     st.markdown(
#         '<h3 style="color: white; direction: rtl;"><i class="bi bi-cpu-fill"></i> پنل مدیریت</h3>',
#         unsafe_allow_html=True,
#     )

#     with st.expander("📊 مانیتورینگ سخت‌افزار", expanded=False):
#         for i in range(torch.cuda.device_count()):
#             used = torch.cuda.memory_reserved(i) / 1024**3
#             st.markdown(
#                 f'<div class="monitor-card"><i class="bi bi-gpu-card"></i> <b>کارت گرافیک {i}</b><br>مصرف: {used:.1f} GB</div>',
#                 unsafe_allow_html=True,
#             )
#         st.markdown(
#             f'<div class="monitor-card" style="background:#1e293b;"><i class="bi bi-cpu"></i> <b>رم سیستم: {psutil.virtual_memory().percent}%</b></div>',
#             unsafe_allow_html=True,
#         )

#     st.divider()
#     files = st.file_uploader("آپلود PDF", type="pdf", accept_multiple_files=True)

#     if files and st.button("🪄 شروع پردازش"):
#         new_docs = []
#         with st.status("🚀 در حال بررسی و پردازش هوشمند...", expanded=True) as status:
#             for f in files:
#                 f_bytes = f.read()
#                 f_hash = hashlib.md5(f_bytes).hexdigest()
#                 meta_path = METADATA_DIR / f"{f_hash}.json"

#                 if meta_path.exists():
#                     st.toast(f"✅ فایل {f.name} قبلاً OCR شده بود.", icon="💾")
#                     with open(meta_path, "r") as m:
#                         combined_text = json.load(m)["full_content"]
#                 else:
#                     status.write(f"🔍 در حال استخراج متن: {f.name}")
#                     imgs = convert_from_bytes(f_bytes, dpi=150)
#                     with ThreadPoolExecutor(max_workers=4) as exe:
#                         texts = list(
#                             exe.map(
#                                 lambda img: pytesseract.image_to_string(
#                                     img, lang="fas+eng"
#                                 ),
#                                 imgs,
#                             )
#                         )
#                     combined_text = clean_ocr_text("\n\n".join(texts))
#                     with open(meta_path, "w") as m:
#                         json.dump({"name": f.name, "full_content": combined_text}, m)
#                     st.toast(f"✨ فایل {f.name} با موفقیت پردازش شد.", icon="✅")

#                 st.session_state.full_texts[f.name] = combined_text
#                 new_docs.append(
#                     Document(page_content=combined_text, metadata={"source": f.name})
#                 )

#             if new_docs:
#                 splits = RecursiveCharacterTextSplitter(
#                     chunk_size=800, chunk_overlap=200
#                 ).split_documents(new_docs)
#                 vs = FAISS.from_documents(splits, embeddings)
#                 vs.index = faiss.index_gpu_to_cpu(vs.index)
#                 vs.save_local(str(INDEX_PATH))
#                 st.session_state.vectorstore = load_vectorstore_on_gpu(
#                     str(INDEX_PATH), embeddings, 1
#                 )
#                 status.update(label="✅ دیتابیس بروزرسانی شد!", state="complete")
#                 st.rerun()

# # --- بخش اصلی گفتگو ---
# st.title("🏢 دستیار هوشمند مدیریت دانش")
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for m in st.session_state.messages:
#     with st.chat_message(m["role"], avatar="👤" if m["role"] == "user" else "🤖"):
#         st.markdown(
#             f'<div style="text-align: right; direction: rtl;">{m["content"]}</div>',
#             unsafe_allow_html=True,
#         )

# if prompt := st.chat_input("سوال خود را بپرسید..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user", avatar="👤"):
#         st.markdown(
#             f'<div style="text-align: right; direction: rtl;">{prompt}</div>',
#             unsafe_allow_html=True,
#         )

#     with st.chat_message("assistant", avatar="🤖"):
#         if st.session_state.vectorstore:
#             with st.spinner("🔍 در حال تحلیل..."):
#                 docs = st.session_state.vectorstore.similarity_search(prompt, k=15)
#                 pairs = [[prompt, d.page_content] for d in docs]
#                 inputs = rerank_t(
#                     pairs,
#                     padding=True,
#                     truncation=True,
#                     return_tensors="pt",
#                     max_length=512,
#                 ).to("cuda:1")

#                 with torch.no_grad():
#                     scores = rerank_m(**inputs).logits.view(-1).float()

#                 context = "\n\n".join(
#                     [
#                         docs[i].page_content
#                         for i in torch.argsort(scores, descending=True)[:8]
#                     ]
#                 )
#                 chain = prompt_template | llm_engine
#                 response = chain.invoke({"context": context, "question": prompt})

#                 ans = (
#                     response.split("پاسخ جامع و تشریحی:")[-1].strip()
#                     if "پاسخ جامع و تشریحی:" in response
#                     else response
#                 )
#                 ans = re.sub(r"\*+", "", ans).strip()

#                 st.markdown(
#                     f'<div style="text-align: right; direction: rtl;">{ans}</div>',
#                     unsafe_allow_html=True,
#                 )
#                 st.session_state.messages.append({"role": "assistant", "content": ans})

#     torch.cuda.empty_cache()
#     gc.collect()


# import streamlit as st
# import torch, gc, json, hashlib, time, uuid, faiss, psutil, os, base64, re
# from pathlib import Path
# from pdf2image import convert_from_bytes
# import pytesseract
# from concurrent.futures import ThreadPoolExecutor
# from vector_manager import load_vectorstore_on_gpu
# from setup_models import setup_llm_and_embeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

# # --- ۱. وارد کردن استایل (فایل style.py شما بدون تغییر می‌ماند) ---
# from style import apply_custom_styles

# # --- ۲. تنظیمات اولیه صفحه (سایدبار پیش‌فرض باز است) ---
# st.set_page_config(
#     page_title="دستیار هوشمند سازمانی", layout="wide", initial_sidebar_state="expanded"
# )

# # --- ۳. اعمال استایل‌ها ---
# apply_custom_styles()


# def clean_ocr_text(text):
#     text = re.sub(r"[°τµ~|—_]{2,}", "", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# DATA_DIR = Path("data")
# INDEX_PATH = DATA_DIR / "vectorstore"
# METADATA_DIR = DATA_DIR / "metadata"
# for p in [METADATA_DIR, INDEX_PATH]:
#     p.mkdir(parents=True, exist_ok=True)

# # لود کردن مدل‌ها با مدیریت حافظه
# if "models" not in st.session_state:
#     with st.spinner("⏳ در حال بیدار کردن مدل‌ها..."):
#         st.session_state.models = setup_llm_and_embeddings()

# embeddings, llm_engine, prompt_template, (rerank_m, rerank_t) = st.session_state.models

# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = load_vectorstore_on_gpu(
#         str(INDEX_PATH), embeddings, 1
#     )

# # --- مدیریت سایدبار ---
# with st.sidebar:
#     st.markdown(
#         '<h3 style="color: white; direction: rtl;"><i class="bi bi-cpu-fill"></i> پنل مدیریت</h3>',
#         unsafe_allow_html=True,
#     )

#     with st.expander("📊 مانیتورینگ سخت‌افزار", expanded=False):
#         for i in range(torch.cuda.device_count()):
#             used = torch.cuda.memory_reserved(i) / 1024**3
#             st.markdown(
#                 f'<div class="monitor-card"><b>GPU {i}</b>: {used:.1f} GB</div>',
#                 unsafe_allow_html=True,
#             )
#         st.markdown(
#             f'<div class="monitor-card" style="background:#1e293b;"><b>RAM</b>: {psutil.virtual_memory().percent}%</div>',
#             unsafe_allow_html=True,
#         )

#     st.divider()
#     files = st.file_uploader("آپلود PDF", type="pdf", accept_multiple_files=True)

#     if files and st.button("🪄 شروع پردازش"):
#         new_docs = []
#         with st.status("🚀 پردازش هوشمند...", expanded=True) as status:
#             for f in files:
#                 f_bytes = f.read()
#                 f_hash = hashlib.md5(f_bytes).hexdigest()
#                 meta_path = METADATA_DIR / f"{f_hash}.json"

#                 if meta_path.exists():
#                     with open(meta_path, "r") as m:
#                         combined_text = json.load(m)["full_content"]
#                 else:
#                     imgs = convert_from_bytes(f_bytes, dpi=150)
#                     with ThreadPoolExecutor(max_workers=4) as exe:
#                         texts = list(
#                             exe.map(
#                                 lambda img: pytesseract.image_to_string(
#                                     img, lang="fas+eng"
#                                 ),
#                                 imgs,
#                             )
#                         )
#                     combined_text = clean_ocr_text("\n\n".join(texts))
#                     with open(meta_path, "w") as m:
#                         json.dump({"name": f.name, "full_content": combined_text}, m)

#                 new_docs.append(
#                     Document(page_content=combined_text, metadata={"source": f.name})
#                 )

#             if new_docs:
#                 splits = RecursiveCharacterTextSplitter(
#                     chunk_size=800, chunk_overlap=200
#                 ).split_documents(new_docs)
#                 vs = FAISS.from_documents(splits, embeddings)
#                 vs.save_local(str(INDEX_PATH))
#                 st.session_state.vectorstore = load_vectorstore_on_gpu(
#                     str(INDEX_PATH), embeddings, 1
#                 )
#                 st.rerun()

# # --- بخش اصلی گفتگو ---
# st.title("🏢 دستیار هوشمند مدیریت دانش")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # نمایش تاریخچه چت (فقط پیام‌های واقعی کاربر و مدل)
# for m in st.session_state.messages:
#     with st.chat_message(m["role"], avatar="👤" if m["role"] == "user" else "🤖"):
#         st.markdown(
#             f'<div style="text-align: right; direction: rtl;">{m["content"]}</div>',
#             unsafe_allow_html=True,
#         )

# if prompt := st.chat_input("سوال خود را بپرسید..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user", avatar="👤"):
#         st.markdown(
#             f'<div style="text-align: right; direction: rtl;">{prompt}</div>',
#             unsafe_allow_html=True,
#         )

#     with st.chat_message("assistant", avatar="🤖"):
#         if st.session_state.vectorstore:
#             with st.spinner("🔍 در حال تحلیل عمیق مستندات..."):
#                 # بازیابی ۱۵ مورد برای دقت در رنکینگ
#                 docs = st.session_state.vectorstore.similarity_search(prompt, k=15)

#                 # مرحله Rerank برای پیدا کردن بهترین قطعات
#                 pairs = [[prompt, d.page_content] for d in docs]
#                 inputs = rerank_t(
#                     pairs,
#                     padding=True,
#                     truncation=True,
#                     return_tensors="pt",
#                     max_length=512,
#                 ).to("cuda:1")
#                 with torch.no_grad():
#                     scores = rerank_m(**inputs).logits.view(-1).float()

#                 # انتخاب ۸ قطعه برتر (مانند کد اول شما برای دقت بالا)
#                 best_indices = torch.argsort(scores, descending=True)[:8]
#                 context = "\n\n".join([docs[i].page_content for i in best_indices])

#                 # اجرای زنجیره پاسخگویی
#                 chain = prompt_template | llm_engine
#                 response = chain.invoke({"context": context, "question": prompt})

#                 # تمیزکاری هوشمند پاسخ (جلوگیری از حذف محتوا و نمایش دستورات سیستم)
#                 ans = response
#                 # اگر مدل تگ‌های اضافی تولید کرد، آن‌ها را فیلتر می‌کنیم
#                 if "پاسخ جامع و تشریحی:" in ans:
#                     ans = ans.split("پاسخ جامع و تشریحی:")[-1]
#                 elif "پاسخ تشریحی:" in ans:
#                     ans = ans.split("پاسخ تشریحی:")[-1]

#                 # حذف نویزهای احتمالی و پاکسازی نهایی
#                 ans = re.sub(
#                     r"System:.*?\n", "", ans, flags=re.DOTALL
#                 )  # حذف دستورات سیستم از خروجی
#                 ans = re.sub(r"Human:.*?\n", "", ans, flags=re.DOTALL)  # حذف تکرار سوال
#                 ans = re.sub(r"\*+", "", ans).strip()

#                 st.markdown(
#                     f'<div style="text-align: right; direction: rtl;">{ans}</div>',
#                     unsafe_allow_html=True,
#                 )
#                 st.session_state.messages.append({"role": "assistant", "content": ans})

#     # آزاد کردن حافظه گرافیکی برای سرعت عمل مستمر
#     torch.cuda.empty_cache()
#     gc.collect()


# import streamlit as st
# import torch, gc, json, hashlib, time, faiss, psutil, os, re
# from pathlib import Path
# from pdf2image import convert_from_bytes
# import pytesseract
# from concurrent.futures import ThreadPoolExecutor
# from vector_manager import load_vectorstore_on_gpu
# from setup_models import setup_llm_and_embeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

# # --- ۱. تنظیمات سیستمی و ظاهری ---
# from style import apply_custom_styles

# st.set_page_config(
#     page_title="Enterprise AI Knowledge Hub",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
# apply_custom_styles()

# # --- ۲. زیرساخت داده‌های مشترک (Global Storage) ---
# DATA_DIR = Path("data")
# INDEX_PATH = DATA_DIR / "vectorstore"
# METADATA_DIR = DATA_DIR / "metadata"
# for p in [METADATA_DIR, INDEX_PATH]:
#     p.mkdir(parents=True, exist_ok=True)


# @st.cache_resource
# def initialize_engine():
#     """بارگذاری مدل‌ها روی GPU برای تمام کاربران"""
#     return setup_llm_and_embeddings()


# models = initialize_engine()
# embeddings, llm_engine, prompt_template, (rerank_m, rerank_t) = models

# # لود کردن ایندکس مرکزی
# if "vectorstore" not in st.session_state:
#     if (INDEX_PATH / "index.faiss").exists():
#         st.session_state.vectorstore = load_vectorstore_on_gpu(
#             str(INDEX_PATH), embeddings, 1
#         )
#     else:
#         st.session_state.vectorstore = None


# # --- ۳. تابع پردازش هوشمند و چک کردن تکرار در کل شبکه ---
# def process_file_globally(file_obj):
#     """بررسی وجود فایل در دیتابیس مرکزی قبل از OCR"""
#     f_bytes = file_obj.read()
#     f_hash = hashlib.md5(f_bytes).hexdigest()
#     meta_path = METADATA_DIR / f"{f_hash}.json"

#     # اگر فایل قبلاً توسط هر کسی آپلود شده باشد
#     if meta_path.exists():
#         with open(meta_path, "r", encoding="utf-8") as m:
#             data = json.load(m)
#             return data["full_content"], True  # فایل موجود است

#     # اگر فایل جدید است، شروع پردازش OCR
#     imgs = convert_from_bytes(f_bytes, dpi=120)
#     with ThreadPoolExecutor() as executor:
#         texts = list(
#             executor.map(
#                 lambda img: pytesseract.image_to_string(img, lang="fas+eng"), imgs
#             )
#         )

#     combined_text = "\n\n".join(texts)
#     # ذخیره در دیتابیس مرکزی برای استفاده بقیه کاربران
#     with open(meta_path, "w", encoding="utf-8") as m:
#         json.dump(
#             {"name": file_obj.name, "full_content": combined_text, "hash": f_hash}, m
#         )

#     return combined_text, False


# # --- ۴. مدیریت آپلود و اعلان‌های توست (Toast) ---
# with st.sidebar:
#     st.markdown("### 💠 مدیریت متمرکز اسناد")
#     uploaded_files = st.file_uploader(
#         "فایل PDF را انتخاب کنید", type="pdf", accept_multiple_files=True
#     )

#     if uploaded_files and st.button("🚀 بررسی و همگام‌سازی", use_container_width=True):
#         new_docs = []
#         with st.status("در حال بررسی وضعیت فایل‌ها در سرور...", expanded=True) as status:
#             for f in uploaded_files:
#                 content, is_already_exists = process_file_globally(f)

#                 if is_already_exists:
#                     # نمایش نوتیفیکیشن توست که فایل در دیتابیس موجود است
#                     st.toast(
#                         f"فایل '{f.name}' قبلاً پردازش شده و در دسترس است.", icon="✅"
#                     )
#                 else:
#                     st.write(f"⏳ فایل جدید شناسایی شد. در حال OCR: {f.name}")
#                     new_docs.append(
#                         Document(page_content=content, metadata={"source": f.name})
#                     )

#             if new_docs:
#                 status.update(label="در حال به‌روزرسانی ایندکس هوشمند...")
#                 splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=750, chunk_overlap=120
#                 )
#                 splits = splitter.split_documents(new_docs)

#                 if st.session_state.vectorstore is None:
#                     st.session_state.vectorstore = FAISS.from_documents(
#                         splits, embeddings
#                     )
#                 else:
#                     st.session_state.vectorstore.add_documents(splits)

#                 st.session_state.vectorstore.save_local(str(INDEX_PATH))
#                 st.success("فایل‌های جدید با موفقیت به دانش سیستم اضافه شدند.")

#             status.update(label="دیتابیس کاملاً همگام است.", state="complete")

# # --- ۵. بخش چت و استریم پاسخ ---
# st.title("🏢 دستیار هوشمند سازمانی")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for m in st.session_state.messages:
#     with st.chat_message(m["role"]):
#         st.markdown(
#             f'<div dir="rtl" style="text-align:right">{m["content"]}</div>',
#             unsafe_allow_html=True,
#         )

# if prompt := st.chat_input("سوال خود را بپرسید..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(
#             f'<div dir="rtl" style="text-align:right">{prompt}</div>',
#             unsafe_allow_html=True,
#         )

#     with st.chat_message("assistant"):
#         if st.session_state.vectorstore:
#             # کانتینر برای استریم کلمه به کلمه
#             resp_container = st.empty()

#             # جستجوی سریع GPU
#             docs = st.session_state.vectorstore.similarity_search(prompt, k=15)

#             # رنکینگ مجدد (Rerank) روی GPU
#             pairs = [[prompt, d.page_content] for d in docs]
#             inputs = rerank_t(
#                 pairs,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512,
#             ).to("cuda:1")

#             with torch.no_grad():
#                 scores = rerank_m(**inputs).logits.view(-1).float()
#                 best_indices = torch.argsort(scores, descending=True)[:7]
#                 context = "\n\n".join([docs[i].page_content for i in best_indices])

#             # تولید پاسخ به صورت Streaming (برای حذف تاخیر ظاهری)
#             full_ans = ""
#             for chunk in llm_engine.stream(
#                 prompt_template.format(context=context, question=prompt)
#             ):
#                 # حذف کلمات سیستمی ناخواسته
#                 chunk = re.sub(
#                     r"(System:|Human:|Assistant:).*", "", chunk, flags=re.DOTALL
#                 )
#                 full_ans += chunk
#                 resp_container.markdown(
#                     f'<div dir="rtl" style="text-align:right">{full_ans} ▌</div>',
#                     unsafe_allow_html=True,
#                 )

#             resp_container.markdown(
#                 f'<div dir="rtl" style="text-align:right">{full_ans}</div>',
#                 unsafe_allow_html=True,
#             )
#             st.session_state.messages.append({"role": "assistant", "content": full_ans})

#             # آزادسازی حافظه
#             torch.cuda.empty_cache()
#             gc.collect()
#         else:
#             st.warning("ابتدا فایل‌های مورد نظر را آپلود یا همگام‌سازی کنید.")


# import os

# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"

# import streamlit as st
# import torch, gc, json, hashlib, time, faiss, psutil, os, re
# from pathlib import Path
# from pdf2image import convert_from_bytes
# import pytesseract
# from concurrent.futures import ThreadPoolExecutor
# from vector_manager import load_vectorstore_on_gpu
# from setup_models import setup_llm_and_embeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

# # --- ۱. تنظیمات و استایل ---
# from style import apply_custom_styles

# st.set_page_config(
#     page_title="Enterprise AI Knowledge Hub",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
# apply_custom_styles()

# # --- ۲. مسیرها و مدل‌ها ---
# DATA_DIR = Path("data")
# INDEX_PATH = DATA_DIR / "vectorstore"
# METADATA_DIR = DATA_DIR / "metadata"
# for p in [METADATA_DIR, INDEX_PATH]:
#     p.mkdir(parents=True, exist_ok=True)


# @st.cache_resource
# def initialize_engine():
#     return setup_llm_and_embeddings()


# models = initialize_engine()
# embeddings, llm_engine, prompt_template, (rerank_m, rerank_t) = models

# if "vectorstore" not in st.session_state:
#     if (INDEX_PATH / "index.faiss").exists():
#         st.session_state.vectorstore = load_vectorstore_on_gpu(
#             str(INDEX_PATH), embeddings, 1
#         )
#     else:
#         st.session_state.vectorstore = None


# # --- ۳. پردازش OCR هوشمند با کش جهانی ---
# def process_file_globally(file_obj):
#     file_obj.seek(0)
#     f_bytes = file_obj.read()
#     f_hash = hashlib.md5(f_bytes).hexdigest()
#     meta_path = METADATA_DIR / f"{f_hash}.json"

#     if meta_path.exists():
#         with open(meta_path, "r", encoding="utf-8") as m:
#             data = json.load(m)
#             return data["full_content"], True

#     imgs = convert_from_bytes(f_bytes, dpi=120)
#     with ThreadPoolExecutor() as executor:
#         texts = list(
#             executor.map(
#                 lambda img: pytesseract.image_to_string(img, lang="fas+eng"), imgs
#             )
#         )

#     combined_text = "\n\n".join(texts)
#     with open(meta_path, "w", encoding="utf-8") as m:
#         json.dump(
#             {"name": file_obj.name, "full_content": combined_text, "hash": f_hash}, m
#         )

#     return combined_text, False


# # --- ۴. سایدبار مدیریتی و مانیتورینگ ---
# with st.sidebar:
#     st.markdown(
#         '<h3 style="color: white; direction: rtl;"><i class="bi bi-cpu-fill"></i> پنل مدیریت</h3>',
#         unsafe_allow_html=True,
#     )

#     with st.expander("📊 مانیتورینگ سخت‌افزار (GPU/RAM)", expanded=True):
#         for i in range(torch.cuda.device_count()):
#             used = torch.cuda.memory_reserved(i) / 1024**3
#             st.markdown(
#                 f'<div class="monitor-card"><b>GPU {i}:</b> {used:.1f} GB</div>',
#                 unsafe_allow_html=True,
#             )
#         st.markdown(
#             f'<div class="monitor-card" style="background:#1e293b;"><b>RAM:</b> {psutil.virtual_memory().percent}%</div>',
#             unsafe_allow_html=True,
#         )

#     st.divider()
#     uploaded_files = st.file_uploader(
#         "فایل PDF (یک یا چندگانه)", type="pdf", accept_multiple_files=True
#     )

#     if uploaded_files and st.button("🚀 همگام‌سازی دیتابیس", use_container_width=True):
#         new_docs = []
#         with st.status("در حال بررسی فایل‌ها...", expanded=True) as status:
#             for f in uploaded_files:
#                 content, is_already_exists = process_file_globally(f)
#                 if is_already_exists:
#                     st.toast(f"فایل در حافظه موجود است: {f.name}", icon="✅")
#                 else:
#                     new_docs.append(
#                         Document(page_content=content, metadata={"source": f.name})
#                     )

#             if new_docs:
#                 splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=1000, chunk_overlap=200
#                 )
#                 splits = splitter.split_documents(new_docs)
#                 if st.session_state.vectorstore is None:
#                     st.session_state.vectorstore = FAISS.from_documents(
#                         splits, embeddings
#                     )
#                 else:
#                     st.session_state.vectorstore.add_documents(splits)
#                 st.session_state.vectorstore.save_local(str(INDEX_PATH))
#             status.update(label="دیتابیس با موفقیت به‌روز شد!", state="complete")

# # --- ۵. چت‌بات با منطق پاسخگویی عمیق (Deep RAG) ---
# st.title("🏢 دستیار هوشمند مدیریت دانش")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for m in st.session_state.messages:
#     with st.chat_message(m["role"]):
#         st.markdown(
#             f'<div dir="rtl" style="text-align:right">{m["content"]}</div>',
#             unsafe_allow_html=True,
#         )

# if prompt := st.chat_input("سوال خود را با جزئیات بپرسید..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(
#             f'<div dir="rtl" style="text-align:right">{prompt}</div>',
#             unsafe_allow_html=True,
#         )

#     with st.chat_message("assistant"):
#         if st.session_state.vectorstore:
#             resp_placeholder = st.empty()

#             # ۱. بازیابی و رنکینگ (بسیار دقیق)
#             docs = st.session_state.vectorstore.similarity_search(prompt, k=15)
#             pairs = [[prompt, d.page_content] for d in docs]
#             inputs = rerank_t(
#                 pairs,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512,
#             ).to("cuda:1")

#             with torch.no_grad():
#                 scores = rerank_m(**inputs).logits.view(-1).float()
#                 # انتخاب ۱۰ قطعه برتر برای داشتن اطلاعات حداکثری
#                 best_indices = torch.argsort(scores, descending=True)[:10]
#                 selected_docs = [docs[i] for i in best_indices]
#                 context_text = "\n\n".join([d.page_content for d in selected_docs])
#                 sources = list(
#                     set([d.metadata.get("source", "ناشناس") for d in selected_docs])
#                 )

#             # ۲. ساخت پرومپت مهندسی شده برای پاسخ طولانی
#             # نکته: اگر کاربر "متن کامل" خواست، مستقیم از متادیتا می‌خوانیم
#             if any(word in prompt for word in ["متن کامل", "کل فایل", "تموم متن"]):
#                 full_raw = ""
#                 for meta_file in METADATA_DIR.glob("*.json"):
#                     with open(meta_file, "r", encoding="utf-8") as m:
#                         data = json.load(m)
#                         full_raw += f"\n--- محتوای فایل: {data['name']} ---\n{data['full_content']}\n"
#                 full_ans = full_raw if full_raw else "متنی پیدا نشد."
#             else:
#                 # دستور صریح به مدل برای پرهیز از پاسخ کوتاه
#                 enhanced_prompt = f"""شما یک کارشناس خبره تحلیل محتوا هستید.
#                 با استفاده از اطلاعات زیر، یک پاسخ **بسیار جامع، مفصل و با تمام جزئیات** به زبان فارسی بنویسید.
#                 تأکید می‌کنم: پاسخ نباید کوتاه باشد. تمام نکات مهم موجود در متن را استخراج و تحلیل کنید.
#                 اگر سوالی پرسیده شده که در متن نیست، بنویسید: 'در مستندات به این مورد اشاره نشده'.

#                 محتوای اسناد:
#                 {context_text}

#                 سوال کاربر: {prompt}

#                 پاسخ تشریحی و کامل:"""

#                 full_ans = ""
#                 for chunk in llm_engine.stream(enhanced_prompt):
#                     # پاکسازی نویزهای خروجی
#                     chunk = re.sub(
#                         r"(System:|Assistant:|Human:|User:).*?",
#                         "",
#                         chunk,
#                         flags=re.IGNORECASE,
#                     )
#                     full_ans += chunk
#                     resp_placeholder.markdown(
#                         f'<p dir="rtl" style="text-align:right">{full_ans} ▌</p>',
#                         unsafe_allow_html=True,
#                     )

#             # ۳. نمایش نهایی همراه با تگ منابع
#             source_html = " ".join(
#                 [
#                     f'<span style="background:#1e293b; padding:2px 8px; border-radius:5px; font-size:12px; margin-right:5px;">📄 {s}</span>'
#                     for s in sources
#                 ]
#             )
#             final_output = f'<div dir="rtl" style="text-align:right;">{full_ans}<br><br><hr>{source_html}</div>'
#             resp_placeholder.markdown(final_output, unsafe_allow_html=True)

#             st.session_state.messages.append({"role": "assistant", "content": full_ans})
#             torch.cuda.empty_cache()
#             gc.collect()
#         else:
#             st.warning(
#                 "دیتابیس خالی است! لطفاً ابتدا فایل‌های خود را در سایدبار آپلود کنید."
#             )


import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import streamlit as st
import torch, gc, json, hashlib, time, faiss, psutil, os, re
from pathlib import Path
from pdf2image import convert_from_bytes
import pytesseract
from concurrent.futures import ThreadPoolExecutor
from vector_manager import load_vectorstore_on_gpu
from setup_models import setup_llm_and_embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- ۱. تنظیمات و استایل ---
from style import apply_custom_styles

st.set_page_config(
    page_title="Enterprise AI Knowledge Hub",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_styles()

# --- ۲. مسیرها و مدل‌ها ---
DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "vectorstore"
METADATA_DIR = DATA_DIR / "metadata"
for p in [METADATA_DIR, INDEX_PATH]:
    p.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def initialize_engine():
    return setup_llm_and_embeddings()


models = initialize_engine()
embeddings, llm_engine, prompt_template, (rerank_m, rerank_t) = models

if "vectorstore" not in st.session_state:
    if (INDEX_PATH / "index.faiss").exists():
        st.session_state.vectorstore = load_vectorstore_on_gpu(
            str(INDEX_PATH), embeddings, 1
        )
    else:
        st.session_state.vectorstore = None


# --- ۳. پردازش OCR هوشمند با کش جهانی ---
def process_file_globally(file_obj):
    file_obj.seek(0)
    f_bytes = file_obj.read()
    f_hash = hashlib.md5(f_bytes).hexdigest()
    meta_path = METADATA_DIR / f"{f_hash}.json"

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as m:
            data = json.load(m)
            return data["full_content"], True

    imgs = convert_from_bytes(f_bytes, dpi=120)
    with ThreadPoolExecutor() as executor:
        texts = list(
            executor.map(
                lambda img: pytesseract.image_to_string(img, lang="fas+eng"), imgs
            )
        )

    combined_text = "\n\n".join(texts)
    with open(meta_path, "w", encoding="utf-8") as m:
        json.dump(
            {"name": file_obj.name, "full_content": combined_text, "hash": f_hash}, m
        )

    return combined_text, False


# --- ۴. سایدبار مدیریتی و مانیتورینگ ---
with st.sidebar:
    st.markdown(
        '<h3 style="color: white; direction: rtl;"><i class="bi bi-cpu-fill"></i> پنل مدیریت</h3>',
        unsafe_allow_html=True,
    )

    with st.expander("📊 مانیتورینگ سخت‌افزار (GPU/RAM)", expanded=True):
        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_reserved(i) / 1024**3
            st.markdown(
                f'<div class="monitor-card"><b>GPU {i}:</b> {used:.1f} GB</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="monitor-card" style="background:#1e293b;"><b>RAM:</b> {psutil.virtual_memory().percent}%</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    uploaded_files = st.file_uploader(
        "فایل PDF (یک یا چندگانه)", type="pdf", accept_multiple_files=True
    )

    #     if uploaded_files and st.button("🚀 همگام‌سازی دیتابیس", use_container_width=True):
    #         new_docs = []
    #         with st.status("در حال بررسی فایل‌ها...", expanded=True) as status:
    #             for f in uploaded_files:
    #                 content, is_already_exists = process_file_globally(f)
    #                 if is_already_exists:
    #                     st.toast(f"فایل در حافظه موجود است: {f.name}", icon="✅")
    #                 else:
    #                     new_docs.append(
    #                         Document(page_content=content, metadata={"source": f.name})
    #                     )

    #             # if new_docs:
    #             #     splitter = RecursiveCharacterTextSplitter(
    #             #         chunk_size=1000, chunk_overlap=200
    #             #     )
    #             #     splits = splitter.split_documents(new_docs)
    #             #     if st.session_state.vectorstore is None:
    #             #         st.session_state.vectorstore = FAISS.from_documents(
    #             #             splits, embeddings
    #             #         )
    #             #     else:
    #             #         st.session_state.vectorstore.add_documents(splits)
    #             #     st.session_state.vectorstore.save_local(str(INDEX_PATH))

    #             # status.update(label="دیتابیس با موفقیت به‌روز شد!", state="complete")

    #             if new_docs:
    #                 splitter = RecursiveCharacterTextSplitter(
    #                     chunk_size=1000, chunk_overlap=200
    #                 )
    #                 splits = splitter.split_documents(new_docs)

    #                 if st.session_state.vectorstore is None:
    #                     st.session_state.vectorstore = FAISS.from_documents(
    #                         splits, embeddings
    #                     )
    #                 else:
    #                     st.session_state.vectorstore.add_documents(splits)

    #                 # --- بخش اصلاح شده برای جلوگیری از RuntimeError ---
    #                 # ۱. گرفتن ایندکس فعلی از GPU
    #                 gpu_index = st.session_state.vectorstore.index

    #                 # ۲. تبدیل موقت به ایندکس CPU برای قابلیت ذخیره‌سازی روی هارد
    #                 st.session_state.vectorstore.index = faiss.index_gpu_to_cpu(gpu_index)

    #                 # ۳. انجام عملیات ذخیره‌سازی
    #                 st.session_state.vectorstore.save_local(str(INDEX_PATH))

    #                 # ۴. بازگرداندن ایندکس به GPU برای حفظ سرعت بالای جستجو در چت
    #                 st.session_state.vectorstore.index = gpu_index
    #                 # --------------------------------------------------

    #             status.update(label="دیتابیس با موفقیت به‌روز شد!", state="complete")

    # # --- ۵. چت‌بات با منطق پاسخگویی عمیق (Deep RAG) ---
    # st.title("🏢 دستیار هوشمند مدیریت دانش")

    if uploaded_files and st.button("🚀 همگام‌سازی دیتابیس", use_container_width=True):
        new_docs = []
        with st.status("در حال بررسی فایل‌ها...", expanded=True) as status:
            for f in uploaded_files:
                content, is_already_exists = process_file_globally(f)

                if is_already_exists:
                    st.toast(f"فایل در حافظه موجود است: {f.name}", icon="✅")

                # اضافه کردن محتوا به لیست (چه جدید باشد چه موجود)
                new_docs.append(
                    Document(page_content=content, metadata={"source": f.name})
                )

            if new_docs:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                splits = splitter.split_documents(new_docs)

                # مدیریت مقداردهی اولیه‌ی دیتابیس
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = FAISS.from_documents(
                        splits, embeddings
                    )
                else:
                    st.session_state.vectorstore.add_documents(splits)

                # --- عملیات ذخیره‌سازی ایمن (انتقال به CPU و بازگشت به GPU) ---
                gpu_index = st.session_state.vectorstore.index
                st.session_state.vectorstore.index = faiss.index_gpu_to_cpu(gpu_index)
                st.session_state.vectorstore.save_local(str(INDEX_PATH))
                st.session_state.vectorstore.index = gpu_index
                # ---------------------------------------------------------

                status.update(label="دیتابیس با موفقیت به‌روز شد!", state="complete")
                st.rerun()
            else:
                status.update(label="فایلی برای پردازش انتخاب نشده است.", state="error")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(
            # f'<div dir="rtl" style="text-align:right">{m["content"]}</div>',
            f'{m["content"]}',
            unsafe_allow_html=True,
        )

if prompt := st.chat_input("سوال خود را با جزئیات بپرسید..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(
            f'<div dir="rtl" style="text-align:right">{prompt}</div>',
            unsafe_allow_html=True,
        )

    with st.chat_message("assistant"):
        if st.session_state.vectorstore:
            resp_placeholder = st.empty()

            # ۱. بازیابی و رنکینگ (بسیار دقیق)
            docs = st.session_state.vectorstore.similarity_search(prompt, k=15)
            pairs = [[prompt, d.page_content] for d in docs]
            inputs = rerank_t(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to("cuda:1")

            with torch.no_grad():
                scores = rerank_m(**inputs).logits.view(-1).float()
                # انتخاب ۱۰ قطعه برتر برای داشتن اطلاعات حداکثری
                best_indices = torch.argsort(scores, descending=True)[:10]
                selected_docs = [docs[i] for i in best_indices]
                context_text = "\n\n".join([d.page_content for d in selected_docs])
                sources = list(
                    set([d.metadata.get("source", "ناشناس") for d in selected_docs])
                )

            # ۲. ساخت پرومپت مهندسی شده برای پاسخ طولانی
            # نکته: اگر کاربر "متن کامل" خواست، مستقیم از متادیتا می‌خوانیم
            if any(word in prompt for word in ["متن کامل", "کل فایل", "تموم متن"]):
                full_raw = ""
                for meta_file in METADATA_DIR.glob("*.json"):
                    with open(meta_file, "r", encoding="utf-8") as m:
                        data = json.load(m)
                        full_raw += f"\n--- محتوای فایل: {data['name']} ---\n{data['full_content']}\n"
                full_ans = full_raw if full_raw else "متنی پیدا نشد."
            else:
                # دستور صریح به مدل برای پرهیز از پاسخ کوتاه
                enhanced_prompt = f"""شما یک کارشناس خبره تحلیل محتوا هستید.
                با استفاده از اطلاعات زیر، یک پاسخ **بسیار جامع، مفصل و با تمام جزئیات** به زبان فارسی بنویسید.
                تأکید می‌کنم: پاسخ نباید کوتاه باشد. تمام نکات مهم موجود در متن را استخراج و تحلیل کنید.
                اگر سوالی پرسیده شده که در متن نیست، بنویسید: 'در مستندات به این مورد اشاره نشده'.

                محتوای اسناد:
                {context_text}

                سوال کاربر: {prompt}
                
                پاسخ تشریحی و کامل:"""

                full_ans = ""
                for chunk in llm_engine.stream(enhanced_prompt):
                    # پاکسازی نویزهای خروجی
                    chunk = re.sub(
                        r"(System:|Assistant:|Human:|User:).*?",
                        "",
                        chunk,
                        flags=re.IGNORECASE,
                    )
                    full_ans += chunk
                    resp_placeholder.markdown(
                        # f'<p dir="rtl" style="text-align:right">{full_ans} ▌</p>',
                        f"{full_ans}",
                        unsafe_allow_html=True,
                    )

            # ۳. نمایش نهایی همراه با تگ منابع
            source_html = " ".join(
                [
                    f'<span style="background:#1e293b; padding:2px 8px; border-radius:5px; font-size:12px; margin-right:5px;">📄 {s}</span>'
                    for s in sources
                ]
            )
            final_output = f'<div dir="rtl" style="text-align:right;">{full_ans}<br><br><hr>{source_html}</div>'
            resp_placeholder.markdown(final_output, unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": full_ans})
            torch.cuda.empty_cache()
            gc.collect()
        else:
            st.warning(
                "دیتابیس خالی است! لطفاً ابتدا فایل‌های خود را در سایدبار آپلود کنید."
            )
