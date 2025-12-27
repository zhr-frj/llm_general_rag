import streamlit as st
import torch
import re
import pytesseract
from pdf2image import convert_from_bytes
from setup_models import setup_llm_and_embeddings, format_docs
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="تحلیلگر هوشمند حرکت", layout="wide")

# ۱. مدیریت تاریخچه چت (جلوگیری از حذف سوالات قبلی)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

try:
    embeddings, llm_engine, prompt_template = setup_llm_and_embeddings()
except Exception as e:
    st.error(f"خطا در لود مدل: {e}")
    st.stop()

def clean_text_pro(text):
    text = text.replace("ی", "ی").replace("ک", "ک")
    f_digits, e_digits = "۰۱۲۳۴۵۶۷۸۹", "0123456789"
    text = text.translate(str.maketrans(f_digits, e_digits))
    text = re.sub(r'[^\u0600-\u06FF\s\d.,;?!()\-]', ' ', text)
    return " ".join(text.split())

def process_high_quality(uploaded_files, _embeddings):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    
    for uploaded_file in uploaded_files:
        with st.spinner(f"در حال اسکن دقیق کتاب (DPI 300)..."):
            images = convert_from_bytes(uploaded_file.read(), dpi=300)
            for i, image in enumerate(images):
                raw_text = pytesseract.image_to_string(image, lang='fas')
                cleaned = clean_text_pro(raw_text)
                
                # تقویت شناسنامه: کلمات کلیدی را به متادیتای صفحات اول اضافه می‌کنیم
                prefix = ""
                if i < 3:
                    prefix = "[اطلاعات شناسنامه: نام کتاب، نویسنده، چاپ، قیمت، تیراژ، ناشر] "
                
                if len(cleaned) > 25:
                    all_docs.append(Document(
                        page_content=prefix + cleaned, 
                        metadata={"page": i+1}
                    ))
    
    vectorstore = FAISS.from_documents(text_splitter.split_documents(all_docs), _embeddings)
    # k=15 برای دقت بالا در مفاهیم
    return vectorstore.as_retriever(search_kwargs={"k": 15})

# --- پنل کناری ---
with st.sidebar:
    st.header("📂 بارگذاری کتاب")
    files = st.file_uploader("فایل PDF", type="pdf", accept_multiple_files=True)
    if st.button("🚀 شروع تحلیل"):
        if files:
            st.session_state.retriever = process_high_quality(files, embeddings)
            st.success("تحلیل با موفقیت انجام شد.")

# ۲. نمایش تاریخچه گفتگو (این بخش مانع پاک شدن سوالات قبلی می‌شود)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ۳. دریافت سوال جدید
if prompt := st.chat_input("سوال خود را بپرسید..."):
    # ذخیره سوال کاربر در تاریخچه
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if st.session_state.retriever:
        with st.chat_message("assistant"):
            with st.spinner("در حال استخراج پاسخ..."):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                docs = st.session_state.retriever.invoke(prompt)
                context = format_docs(docs)
                
                chain = (
                    {"context": lambda x: context, "question": RunnablePassthrough()} 
                    | prompt_template | llm_engine | StrOutputParser()
                )
                
                response = chain.invoke(prompt)
                pages = ", ".join(set([str(d.metadata['page']) for d in docs]))
                full_res = f"{response}\n\n*📍 منابع:* صفحات {pages}"
                
                st.markdown(full_res)
                # ذخیره پاسخ سیستم در تاریخچه
                st.session_state.messages.append({"role": "assistant", "content": full_res})

if torch.cuda.is_available():
    torch.cuda.empty_cache()
