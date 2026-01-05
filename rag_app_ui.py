import streamlit as st
import torch
import re
import pytesseract
import gc
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_bytes
from setup_models import setup_llm_and_embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================
# مسیرها و پوشه‌های داده برای OCR و metadata
# ==========================
from pathlib import Path
import datetime
import json

DATA_DIR = Path("data")
OCR_DIR = DATA_DIR / "ocr_texts"
METADATA_DIR = DATA_DIR / "metadata"

OCR_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(page_title="سامانه مرکزی تحلیل اسناد", layout="wide", page_icon="🏢")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "full_raw_text" not in st.session_state:
    st.session_state.full_raw_text = []

embeddings, llm_engine, prompt_template = setup_llm_and_embeddings()








# def load_existing_documents(_embeddings):
#     all_docs = []
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     st.session_state.full_raw_text = []

#     for metadata_file in METADATA_DIR.glob("*.json"):
#         with open(metadata_file, "r", encoding="utf-8") as f:
#             metadata = json.load(f)
#         ocr_path = Path(metadata["ocr_text_path"])
#         if ocr_path.exists():
#             with open(ocr_path, "r", encoding="utf-8") as f:
#                 text = f.read()
#             st.session_state.full_raw_text.append(f"--- {metadata['original_filename']} ---\n{text}")
#             all_docs.append(Document(page_content=text, metadata={"filename": metadata['original_filename']}))

#     if all_docs:
#         vectorstore = FAISS.from_documents(text_splitter.split_documents(all_docs), _embeddings)
#         return vectorstore.as_retriever(search_kwargs={"k": 10})
#     return None





def load_existing_documents(_embeddings):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # پاک‌سازی و بازسازی state ها
    st.session_state.full_raw_text = []
    st.session_state.metadata_text = []

    for metadata_file in METADATA_DIR.glob("*.json"):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # 🔑 بازیابی متادیتای شناسنامه‌ای ذخیره‌شده
        if "book_metadata_text" in metadata:
            st.session_state.metadata_text.extend(metadata["book_metadata_text"])

        ocr_path = Path(metadata["ocr_text_path"])
        if ocr_path.exists():
            with open(ocr_path, "r", encoding="utf-8") as f:
                text = f.read()

            # متن کامل برای fallback و درخواست «کل متن»
            st.session_state.full_raw_text.append(
                f"--- {metadata['original_filename']} ---\n{text}"
            )

            # chunking برای RAG
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "filename": metadata["original_filename"],
                            "chunk_id": i
                        }
                    )
                )

    if all_docs:
        vectorstore = FAISS.from_documents(all_docs, _embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 20})

    return None




# بارگذاری فایل‌های موجود در دیتابیس هنگام شروع برنامه
if st.session_state.retriever is None:
    st.session_state.retriever = load_existing_documents(embeddings)











def clean_text_pro(text):
    text = text.replace("ی", "ی").replace("ک", "ک")
    text = re.sub(r'[^\u0600-\u06FF\s\d.,;?!()\-]', ' ', text)
    return " ".join(text.split())

def process_single_page(args):
    idx, image = args
    raw_text = pytesseract.image_to_string(image, lang='fas')
    return idx + 1, clean_text_pro(raw_text)

def process_high_quality(uploaded_files, _embeddings):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.full_raw_text = []
    
    for uploaded_file in uploaded_files:
        with st.spinner(f"در حال نمایه‌سازی سند: {uploaded_file.name}"):
            images = convert_from_bytes(uploaded_file.read(), dpi=200)
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(process_single_page, enumerate(images)))
            
            results.sort(key=lambda x: x[0])
            for page_num, text in results:
                # ذخیره متن خام برای درخواست‌های «کل متن» بدون دخالت هوش مصنوعی
                st.session_state.full_raw_text.append(f"--- صفحه {page_num} ---\n{text}")
                all_docs.append(Document(page_content=text, metadata={"page": page_num}))
            gc.collect()
    
    vectorstore = FAISS.from_documents(text_splitter.split_documents(all_docs), _embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 20})








# ==========================
# نسخه جدید تابع OCR و ذخیره‌سازی فایل‌ها
# ==========================
def process_high_quality_v2(uploaded_files, _embeddings):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.full_raw_text = []

    def process_single_page_inner(args):
        idx, image = args
        raw_text = pytesseract.image_to_string(image, lang='fas')
        text = raw_text.replace("ی", "ی").replace("ک", "ک")
        import re
        return idx + 1, " ".join(re.sub(r'[^\u0600-\u06FF\s\d.,;?!()\-]', ' ', text).split())

    for uploaded_file in uploaded_files:
        original_name = uploaded_file.name
        today = datetime.date.today().strftime("%Y%m%d")
        base_filename = f"{today}{original_name.replace(' ', '')}"
        ocr_path = OCR_DIR / f"{base_filename}.txt"
        metadata_path = METADATA_DIR / f"{base_filename}.json"
        # 🔑 کلمات راهنما برای اطلاعات شناسنامه‌ای
        META_HINTS = ["شابک", "ISBN", "انتشارات", "ناشر", "چاپ", "قیمت", "ریال", "تومان"]
        metadata_texts = []

        # اگر فایل قبلاً OCR شده، استفاده کن
        if metadata_path.exists() and ocr_path.exists():
            st.info(f"فایل {original_name} از قبل OCR شده است. استفاده از نسخه موجود.")
            with open(ocr_path, "r", encoding="utf-8") as f:
                text = f.read()
            st.session_state.full_raw_text.append(f"--- {original_name} ---\n{text}")
            all_docs.append(Document(page_content=text, metadata={"filename": original_name}))
            continue

        # OCR جدید
        with st.spinner(f"در حال نمایه‌سازی سند: {original_name}"):
            images = convert_from_bytes(uploaded_file.read(), dpi=200)
            all_text_pages = []

            from concurrent.futures import ThreadPoolExecutor
            import gc

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(process_single_page_inner, enumerate(images)))

            results.sort(key=lambda x: x[0])
            for page_num, page_text in results:
                st.session_state.full_raw_text.append(f"--- صفحه {page_num} ---\n{page_text}")
                all_text_pages.append(page_text)
                all_docs.append(Document(page_content=page_text, metadata={"filename": original_name, "page": page_num}))
                # 🔍 استخراج اطلاعات شناسنامه‌ای
                if any(k in page_text for k in META_HINTS):
                    metadata_texts.append(page_text)

            # ذخیره متن OCR
            with open(ocr_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(all_text_pages))

            # ذخیره metadata
            metadata = {
                "original_filename": original_name,
                "ocr_text_path": str(ocr_path),
                "upload_date": today,
                "num_pages": len(all_text_pages),
                "book_metadata_text": metadata_texts
            }
            
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            # 🔑 ذخیره متادیتای شناسنامه‌ای در session_state
            if "metadata_text" not in st.session_state:
                st.session_state.metadata_text = []

            st.session_state.metadata_text.extend(metadata_texts)

            gc.collect()

    # ساخت vectorstore
    vectorstore = FAISS.from_documents(text_splitter.split_documents(all_docs), _embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 20})








# --- UI ---
st.title("🏢 سامانه مرکزی استخراج دانش و تحلیل اسناد")
st.caption("نسخه نهایی داینامیک - وفاداری مطلق به متن")

with st.sidebar:
    st.header("بارگذاری اسناد")
    uploaded_files = st.file_uploader("فایل PDF را انتخاب کنید", type="pdf", accept_multiple_files=True)
    
    
    # if uploaded_files and st.button("شروع تحلیل"):
    #     # st.session_state.retriever = process_high_quality(uploaded_files, embeddings)
    #     st.session_state.retriever = process_high_quality_v2(uploaded_files, embeddings)
        
    #     st.success("فرآیند با موفقیت انجام شد.")
    #     st.rerun()
        
        
        
    if uploaded_files and st.button("شروع تحلیل"):
    # OCR و ذخیره‌سازی فایل‌های جدید
        process_high_quality_v2(uploaded_files, embeddings)
    # بازسازی retriever با استفاده از همه فایل‌ها (جدید و قبلی)
        st.session_state.retriever = load_existing_documents(embeddings)
        st.success("فرآیند با موفقیت انجام شد.")
        st.rerun()
        
        
        
        
        
        
        
        
        

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt := st.chat_input("سوال خود را بپرسید..."):
    if st.session_state.retriever:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            
            # ۱. پاسخ مستقیم به درخواست «کل متن» (بدون استفاده از مدل برای جلوگیری از توهم)
            if any(x in prompt for x in ["کل متن", "تمام متن", "متن کامل"]):
                full_res = "### متن استخراج شده از کل داکیومنت:\n\n" + "\n\n".join(st.session_state.full_raw_text)
                placeholder.markdown(full_res)
            
            # ۲. پاسخگویی مبتنی بر RAG
            else:
                full_res = ""
                # تشخیص سوالات شناسنامه‌ای برای اولویت‌بندی صفحات اول                                                   
                is_meta = any(k in prompt for k in ["نویسنده", "قیمت", "ناشر", "تیراژ", "چاپ", "شابک", "ISBN"])

                if is_meta and "metadata_text" in st.session_state and st.session_state.metadata_text:
                    # 🔑 پاسخ دقیق فقط از متادیتای شناسنامه‌ای
                    context = "\n\n".join(st.session_state.metadata_text)
                else:
                    # 🔍 RAG معمولی
                    docs = st.session_state.retriever.invoke(prompt)
                    context = "\n\n".join([d.page_content for d in docs])

                    # 🔁 fallback اگر بازیابی ضعیف بود
                    if len(context.strip()) < 800:
                        context = "\n".join(st.session_state.full_raw_text)
                    
                    

                # if context.strip():
                #     try:
                #         chain = ({"context": lambda x: context, "question": RunnablePassthrough()} | prompt_template | llm_engine | StrOutputParser())
                #         for chunk in chain.stream(prompt):
                #             full_res += chunk
                #             placeholder.markdown(full_res + "▌")
                #     except Exception:
                #         full_res = "⚠️ خطای فنی در تحلیل متن. لطفاً دوباره تلاش کنید."
                # else:
                #     full_res = "در اسناد بارگذاری شده اطلاعاتی درباره این موضوع یافت نشد."



                if context.strip():
                    try:
                        chain = ({"context": lambda x: context, "question": RunnablePassthrough()} | prompt_template | llm_engine | StrOutputParser())
                        for chunk in chain.stream(prompt):
                            full_res += chunk
                            placeholder.markdown(full_res + "▌")
                    except Exception:
                        full_res = "⚠️ خطای فنی در تحلیل متن. لطفاً دوباره تلاش کنید."
                else:
                    full_res = "ℹ️ اطلاعات مرتبط با این سوال در اسناد موجود یافت نشد."
    
    
    
    
                
                placeholder.markdown(full_res)
            
            st.session_state.messages.append({"role": "assistant", "content": full_res})

if torch.cuda.is_available(): torch.cuda.empty_cache()




