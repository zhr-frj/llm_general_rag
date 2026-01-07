import streamlit as st
import torch
import re
import pytesseract
import gc
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_bytes
from setup_models import setup_llm_and_embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path
import datetime
import json

# ==========================
# مسیرها و پوشه‌ها
# ==========================
DATA_DIR = Path("data")
OCR_DIR = DATA_DIR / "ocr_texts"
METADATA_DIR = DATA_DIR / "metadata"

OCR_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="سامانه مرکزی تحلیل اسناد",
    layout="wide",
    page_icon="🏢"
)

# ==========================
# session state ها
# ==========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "full_raw_text" not in st.session_state:
    st.session_state.full_raw_text = []

if "metadata_text" not in st.session_state:
    st.session_state.metadata_text = []

if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = set()

# ==========================
# مدل‌ها
# ==========================
embeddings, llm_engine, prompt_template = setup_llm_and_embeddings()

# ==========================
# 🔑 تابع جدید: دریافت «کل متن» بدون LLM و RAG
# ==========================
def get_full_documents_text():
    texts = []
    for metadata_file in METADATA_DIR.glob("*.json"):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        ocr_path = Path(metadata["ocr_text_path"])
        if ocr_path.exists():
            with open(ocr_path, "r", encoding="utf-8") as f:
                text = f.read()
            texts.append(
                f"--- {metadata['original_filename']} ---\n{text}"
            )

    return "\n\n".join(texts)

# ==========================
# بارگذاری اسناد قبلی
# ==========================
def load_existing_documents(_embeddings):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    st.session_state.full_raw_text = []
    st.session_state.metadata_text = []

    for metadata_file in METADATA_DIR.glob("*.json"):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if "book_metadata_text" in metadata:
            st.session_state.metadata_text.extend(metadata["book_metadata_text"])

        ocr_path = Path(metadata["ocr_text_path"])
        if not ocr_path.exists():
            continue

        with open(ocr_path, "r", encoding="utf-8") as f:
            text = f.read()

        st.session_state.full_raw_text.append(
            f"--- {metadata['original_filename']} ---\n{text}"
        )

        for i, chunk in enumerate(splitter.split_text(text)):
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
        vs = FAISS.from_documents(all_docs, _embeddings)
        return vs.as_retriever(search_kwargs={"k": 8})

    return None


if st.session_state.retriever is None:
    st.session_state.retriever = load_existing_documents(embeddings)

# ==========================
# OCR utils
# ==========================
def clean_text(text):
    text = text.replace("ی", "ی").replace("ک", "ک")
    text = re.sub(r'[^\u0600-\u06FF\s\d.,;?!()\-]', ' ', text)
    return " ".join(text.split())

def process_single_page(args):
    idx, image = args
    raw = pytesseract.image_to_string(image, lang="fas")
    return idx + 1, clean_text(raw)

# ==========================
# OCR + ذخیره‌سازی امن
# ==========================
def process_high_quality_v2(uploaded_files, _embeddings):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.full_raw_text = []

    META_HINTS = ["شابک", "ISBN", "انتشارات", "ناشر", "چاپ", "قیمت", "ریال", "تومان"]

    for uploaded_file in uploaded_files:
        original_name = uploaded_file.name
        today = datetime.date.today().strftime("%Y%m%d")

        file_bytes = uploaded_file.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        uploaded_file.seek(0)

        if file_hash in st.session_state.processed_hashes:
            st.info("این فایل قبلاً در این نشست پردازش شده است.")
            continue

        st.session_state.processed_hashes.add(file_hash)

        base_filename = f"{file_hash}_{original_name.replace(' ', '')}"
        ocr_path = OCR_DIR / f"{base_filename}.txt"
        metadata_path = METADATA_DIR / f"{base_filename}.json"
        lock_path = METADATA_DIR / f"{base_filename}.lock"

        if lock_path.exists():
            st.warning("⏳ این فایل در حال پردازش توسط کاربر دیگری است.")
            continue

        if metadata_path.exists() and ocr_path.exists():
            st.info(f"فایل {original_name} قبلاً OCR شده است.")
            continue

        lock_path.touch()

        try:
            with st.spinner(f"در حال OCR: {original_name}"):
                images = convert_from_bytes(uploaded_file.read(), dpi=200)
                pages = []

                with ThreadPoolExecutor(max_workers=4) as ex:
                    results = list(ex.map(process_single_page, enumerate(images)))

                results.sort(key=lambda x: x[0])
                meta_texts = []

                for page_num, page_text in results:
                    pages.append(page_text)
                    all_docs.append(
                        Document(
                            page_content=page_text,
                            metadata={"filename": original_name, "page": page_num}
                        )
                    )
                    if any(k in page_text for k in META_HINTS):
                        meta_texts.append(page_text)

                with open(ocr_path, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(pages))

                metadata = {
                    "original_filename": original_name,
                    "ocr_text_path": str(ocr_path),
                    "upload_date": today,
                    "num_pages": len(pages),
                    "book_metadata_text": meta_texts
                }

                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                st.session_state.metadata_text.extend(meta_texts)

        finally:
            if lock_path.exists():
                lock_path.unlink()
            gc.collect()

    if all_docs:
        vs = FAISS.from_documents(all_docs, _embeddings)
        return vs.as_retriever(search_kwargs={"k": 8})

    return None

# ==========================
# UI
# ==========================
st.title("🏢 سامانه مرکزی استخراج دانش و تحلیل اسناد")
st.caption("نسخه پایدار چندکاربره")

with st.sidebar:
    uploaded_files = st.file_uploader(
        "فایل PDF را انتخاب کنید",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.button("شروع تحلیل"):
        process_high_quality_v2(uploaded_files, embeddings)
        st.session_state.retriever = load_existing_documents(embeddings)
        st.success("پردازش انجام شد.")
        st.rerun()

# ==========================
# Chat
# ==========================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("سوال خود را بپرسید..."):
    if not st.session_state.retriever:
        st.warning("هیچ سندی بارگذاری نشده است.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()

            # 🔑 شرط جدید: درخواست «کل متن»
            if any(x in prompt for x in ["کل متن", "تمام متن", "متن کامل"]):
                full_text = get_full_documents_text()
                if not full_text.strip():
                    placeholder.markdown("ℹ️ هنوز متنی برای نمایش وجود ندارد.")
                else:
                    placeholder.markdown("### 📄 متن کامل اسناد:\n\n" + full_text)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_text
                })
                st.stop()  

            full_res = ""
            is_meta = any(k in prompt for k in ["نویسنده", "ناشر", "قیمت", "شابک", "ISBN"])

            if is_meta and st.session_state.metadata_text:
                context = "\n\n".join(st.session_state.metadata_text)
            else:
                docs = st.session_state.retriever.invoke(prompt)
                context = "\n\n".join(d.page_content for d in docs)

            context = context[:6000]

            try:
                chain = (
                    {"context": lambda _: context, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm_engine
                    | StrOutputParser()
                )
                for chunk in chain.stream(prompt):
                    full_res += chunk
                    placeholder.markdown(full_res + "▌")
            except Exception:
                full_res = "⚠️ خطا در پردازش پاسخ."

            placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})

if torch.cuda.is_available():
    torch.cuda.empty_cache()