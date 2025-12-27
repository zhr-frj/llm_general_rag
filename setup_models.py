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

# تنظیمات کوانتایزیشن برای جلوگیری از کرش
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)

SYSTEM_TEMPLATE = """<start_of_turn>system
شما دستیار هوشمند کتاب هستید. بر اساس متن ارائه شده پاسخ دهید.
۱. مفهوم محور باشید: حتی اگر کلمات دقیقاً یکی نبودند، بر اساس محتوا جواب دهید.
۲. دقیق باشید: اطلاعات شناسنامه‌ای را از صفحات اول استخراج کنید.
۳. اگر پاسخ در متن نیست، بنویسید "در اسناد یافت نشد".
<end_of_turn>"""

HUMAN_TEMPLATE = """<start_of_turn>user
متن کتاب:
{context}

سوال: {question}<end_of_turn>
<start_of_turn>model
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("human", HUMAN_TEMPLATE)
])

def format_docs(docs):
    return "\n\n".join(f"[صفحه {doc.metadata.get('page')}]: {doc.page_content}" for doc in docs)

@st.cache_resource
def setup_llm_and_embeddings():
    warnings.filterwarnings("ignore")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        quantization_config=bnb_config,
        offload_folder="offload"
    )
    
    text_gen_pipeline = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_new_tokens=600,
        do_sample=False, 
        temperature=0.0,
        return_full_text=False
    )
    
    return embeddings, HuggingFacePipeline(pipeline=text_gen_pipeline), PROMPT_TEMPLATE
