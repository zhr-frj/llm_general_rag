import os
import torch
import warnings
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, AutoModelForSequenceClassification
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# ۱. تنظیمات سیستمی برای آزادسازی گلوگاه دیسک و پردازنده
os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = "./models/gemma_model"
EMBEDDING_PATH = "./models/e5_model"
RERANKER_PATH = "./models/bge_reranker"

# ۲. پیکربندی کوانتیزاسیون برای بهینه‌سازی VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# ۳. تعریف قالب‌های چت (Prompt Templates) - سازگار با Gemma 2
SYSTEM_TEMPLATE = """<start_of_turn>system
شما یک تحلیلگر ارشد اسناد هستید که با دقت بالا و بر اساس مستندات ارائه شده پاسخ می‌دهید.<end_of_turn>"""

HUMAN_TEMPLATE = """<start_of_turn>user
بافتار مستندات:
{context}

سوال کاربر:
{question}<end_of_turn>
<start_of_turn>model
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("human", HUMAN_TEMPLATE)
])

@st.cache_resource
def setup_llm_and_embeddings():
    warnings.filterwarnings("ignore")

    # ۴. مدیریت هوشمند حافظه برای ۲ کارت گرافیک RTX A5000
    # رزرو فضای کافی برای کاربران همزمان (KV-Cache)
    max_memory_map = {0: "20GiB", 1: "20GiB"} 

    # ۵. بارگذاری سریع توکن‌ساز (Fast Tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)

    # ۶. بارگذاری مدل اصلی با تکنیک Direct-to-GPU و حذف گلوگاه SSD
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto",              # توزیع هوشمند بین GPUها
        max_memory=max_memory_map,
        quantization_config=bnb_config,
        trust_remote_code=True,
        local_files_only=True,

        # بهینه‌سازی‌های سرعت و حافظه
        attn_implementation="sdpa",     # استفاده از Flash Attention برای سرعت پاسخگویی
        torch_dtype=torch.float16,
        use_cache=True,                 # برای افزایش سرعت در چت‌های طولانی
        low_cpu_mem_usage=True,         # لود مستقیم و جلوگیری از اشغال RAM سیستم
        use_safetensors=True,           # لود موازی و سریع وزن‌ها
        offload_state_dict=True         # مدیریت بهینه انتقال داده از دیسک
    )

    # ۷. ایجاد پایپ‌لاین تولید متن
    text_gen_pipeline = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_new_tokens=1024,            # ظرفیت پاسخ‌دهی طولانی برای کاربران
        temperature=0.1,
        do_sample=True,
        batch_size=4                    # آمادگی برای پردازش موازی درخواست‌ها
    )

    # ۸. بارگذاری مدل‌های جستجو روی GPU دوم (تفکیک بار کاری)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_PATH,
        model_kwargs={'device': 'cuda:1'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # ۹. بارگذاری مدل رنکر (Reranker)
    rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH, local_files_only=True)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_PATH, local_files_only=True)
    rerank_model.to("cuda:1").eval()

    return embeddings, HuggingFacePipeline(pipeline=text_gen_pipeline), PROMPT_TEMPLATE, (rerank_model, rerank_tokenizer)

