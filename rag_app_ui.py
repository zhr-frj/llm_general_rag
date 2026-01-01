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

st.set_page_config(page_title="سامانه مرکزی تحلیل اسناد", layout="wide", page_icon="🏢")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "full_raw_text" not in st.session_state:
    st.session_state.full_raw_text = []

embeddings, llm_engine, prompt_template = setup_llm_and_embeddings()

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
    return vectorstore.as_retriever(search_kwargs={"k": 10})

# --- UI ---
st.title("🏢 سامانه مرکزی استخراج دانش و تحلیل اسناد")
st.caption("نسخه نهایی داینامیک - وفاداری مطلق به متن")

with st.sidebar:
    st.header("بارگذاری اسناد")
    uploaded_files = st.file_uploader("فایل PDF را انتخاب کنید", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("شروع تحلیل"):
        st.session_state.retriever = process_high_quality(uploaded_files, embeddings)
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
                is_meta = any(k in prompt for k in ["نویسنده", "قیمت", "ناشر", "تیراژ", "چاپ", "مشخصات"])
                
                if is_meta:
                    # برای مشخصات، صفحات ابتدایی را به عنوان اولویت اول بفرست
                    context = "\n".join(st.session_state.full_raw_text[:5])
                else:
                    # برای سایر سوالات، جستجوی معنایی انجام بده
                    docs = st.session_state.retriever.invoke(prompt)
                    context = "\n\n".join([d.page_content for d in docs])

                if context.strip():
                    try:
                        chain = ({"context": lambda x: context, "question": RunnablePassthrough()} | prompt_template | llm_engine | StrOutputParser())
                        for chunk in chain.stream(prompt):
                            full_res += chunk
                            placeholder.markdown(full_res + "▌")
                    except Exception:
                        full_res = "⚠️ خطای فنی در تحلیل متن. لطفاً دوباره تلاش کنید."
                else:
                    full_res = "در اسناد بارگذاری شده اطلاعاتی درباره این موضوع یافت نشد."
                
                placeholder.markdown(full_res)
            
            st.session_state.messages.append({"role": "assistant", "content": full_res})

if torch.cuda.is_available(): torch.cuda.empty_cache()
