import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

MODEL_NAME = "unsloth/gemma-2-9b-it"
embedding_name = 'sentence-transformers/LaBSE'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# پرامپت فوق امنیتی برای جلوگیری از توهم و پاسخ به سوالات متفرقه
SYSTEM_TEMPLATE = """<start_of_turn>system
شما یک سیستم استخراج دانش سازمانی هستید که فقط بر اساس مستندات ارائه شده پاسخ می‌دهد.
قوانین غیرقابل تغییر:
۱. فقط و فقط به زبان فارسی پاسخ دهید.
۲. اگر پاسخ در "بافتار متن" (Context) وجود ندارد، یا سوال درباره اطلاعات عمومی (مثل پایتخت‌ها، تاریخ، یا دانستنی‌ها) است، دقیقاً بنویسید: "در اسناد بارگذاری شده اطلاعاتی درباره این موضوع یافت نشد."
۳. به هیچ عنوان از دانش درونی خود برای پاسخگویی استفاده نکنید.
۴. حق ندارید متن را تخیل کنید یا داستان‌سرایی کنید.
<end_of_turn>"""

HUMAN_TEMPLATE = """<start_of_turn>user
بافتار متن (مبنای پاسخگویی):
{context}

پرسش کاربر: {question}<end_of_turn>
<start_of_turn>model
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("human", HUMAN_TEMPLATE)
])

@st.cache_resource
def setup_llm_and_embeddings():
    warnings.filterwarnings("ignore")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    text_gen_pipeline = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_new_tokens=1500,
        temperature=0.0,       # صفر کردن خلاقیت برای جلوگیری از توهم (Hallucination)
        repetition_penalty=1.1,
        do_sample=False,       # اجبار مدل به وفاداری ۱۰۰ درصدی به متن
    )
    return embeddings, HuggingFacePipeline(pipeline=text_gen_pipeline), PROMPT_TEMPLATE	
