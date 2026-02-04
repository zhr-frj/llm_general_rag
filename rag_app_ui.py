import streamlit as st
import torch
import hashlib
import os
import json
import re
from pathlib import Path
from pdf2image import convert_from_bytes
import pytesseract
from concurrent.futures import ThreadPoolExecutor

# وارد کردن توابع از دو فایل دیگر
from setup_models import setup_llm_and_embeddings
from vector_manager import load_vectorstore_on_gpu, search_documents # <--- اضافه شد

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# تنظیمات مسیرها
os.environ['TESSDATA_PREFIX'] = os.path.abspath("./models/")
DATA_DIR = Path("data")
OCR_DIR = DATA_DIR / "ocr_texts"
METADATA_DIR = DATA_DIR / "metadata"
INDEX_PATH = "models/faiss_index" # مسیر ذخیره دیتابیس روی هارد
OCR_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="سامانه تحلیل اسناد Enterprise", layout="wide")

# بارگذاری مدل‌ها
embeddings, llm_engine, prompt_template, (rerank_model, rerank_tokenizer) = setup_llm_and_embeddings()

# --- توابع سیستمی ---

def clean_text(text):
    text = text.replace("ی", "ی").replace("ک", "ک")
    text = re.sub(r'[^\u0600-\u06FF\s\d.,;?!()\-]', ' ', text)
    return " ".join(text.split())

def process_single_page(args):
    idx, image = args
    raw = pytesseract.image_to_string(image, lang="fas")
    return idx + 1, clean_text(raw)

def index_documents_from_disk(_embeddings):
    """اسکن هارد، ساخت ایندکس و انتقال به GPU از طریق vector_manager"""
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=250)

    meta_files = list(METADATA_DIR.glob("*.json"))
    if not meta_files: return None

    for meta_file in meta_files:
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ocr_path = Path(meta["ocr_text_path"])
            if ocr_path.exists():
                with open(ocr_path, "r", encoding="utf-8") as f:
                    text = f.read()
                if text.strip():
                    for chunk in splitter.split_text(text):
                        all_docs.append(Document(page_content=chunk, metadata={"source": meta["original_filename"]}))
        except: continue

    if all_docs:
        # ۱. ساخت دیتابیس موقت در رم
        vs = FAISS.from_documents(all_docs, _embeddings)
        # ۲. ذخیره روی هارد برای استفاده‌های بعدی
        vs.save_local(INDEX_PATH)
        # ۳. استفاده از فایل vector_manager برای انتقال به GPU 1 و پاکسازی حافظه
        vs_gpu = load_vectorstore_on_gpu(INDEX_PATH, _embeddings)
        return vs_gpu
    return None

def apply_reranking(query, documents):
    if not documents: return []
    pairs = [[query, doc.page_content] for doc in documents]
    device = next(rerank_model.parameters()).device
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        scores = rerank_model(**inputs).logits.view(-1,).float()
        combined = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in combined[:8]]

# --- شروع منطق اصلی برنامه ---

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = index_documents_from_disk(embeddings)

st.title("🏢 سامانه هوشمند استخراج دانش")

with st.sidebar:
    st.subheader("مدیریت اسناد")
    uploaded_files = st.file_uploader("آپلود PDF", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("تحلیل و ایندکس گذاری"):
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
            base_name = f"{file_hash}_{uploaded_file.name.replace(' ', '')}"
            ocr_path = OCR_DIR / f"{base_name}.txt"

            if not ocr_path.exists():
                with st.spinner(f"🔄 در حال OCR: {uploaded_file.name}"):
                    images = convert_from_bytes(file_bytes, dpi=200)
                    with ThreadPoolExecutor(max_workers=2) as ex:
                        results = list(ex.map(process_single_page, enumerate(images)))
                    results.sort(key=lambda x: x[0])
                    full_text = "\n\n".join([r[1] for r in results])
                    with open(ocr_path, "w", encoding="utf-8") as f: f.write(full_text)
                    with open(METADATA_DIR / f"{base_name}.json", "w", encoding="utf-8") as f:
                        json.dump({"original_filename": uploaded_file.name, "ocr_text_path": str(ocr_path)}, f)
        
        st.session_state.vectorstore = index_documents_from_disk(embeddings)
        st.rerun()

    if st.button("پاکسازی گفتگو"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("سوال خود را بپرسید..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = index_documents_from_disk(embeddings)

        if st.session_state.vectorstore:
            placeholder = st.empty()
            # استفاده از تابع جستجو در vector_manager (با حفظ تنظیمات قبلی k=20)
            raw_docs = st.session_state.vectorstore.similarity_search(prompt, k=20)
            final_docs = apply_reranking(prompt, raw_docs)

            context = "\n\n".join(d.page_content for d in final_docs)
            chain = (
                {"context": lambda _: context, "question": RunnablePassthrough()}
                | prompt_template | llm_engine | StrOutputParser()
            )

            full_res = ""
            for chunk in chain.stream(prompt):
                full_res += chunk
                placeholder.markdown(full_res + "▌")
            placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
        else:
            st.error("❌ دیتابیس خالی است.")

if torch.cuda.is_available(): torch.cuda.empty_cache()