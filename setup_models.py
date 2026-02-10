import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    AutoModelForSequenceClassification,
)
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = "./models/gemma_model"
EMBEDDING_PATH = "./models/e5_model"
RERANKER_PATH = "./models/bge_reranker"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

# اصلاح پرامپت برای هدایت مدل به پاسخ‌دهی تشریحی
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """شما یک دستیار هوشمند خبره هستید. 
    وظیفه شما پاسخگویی کامل و تشریحی به سوالات بر اساس 'بافتار' (Context) ارائه شده است.
    نکات مهم:
    1. فقط و فقط بر اساس متن ارائه شده پاسخ دهید.
    2. اگر پاسخ در متن نبود، بگویید 'اطلاعاتی در مستندات یافت نشد'.
    3. به هیچ وجه سوال کاربر را در ابتدای پاسخ تکرار نکنید. مستقیم پاسخ را بنویسید.""",
        ),
        ("human", "بافتار:\n{context}\n\nسوال: {question}\n\nپاسخ تشریحی:"),
    ]
)


@st.cache_resource
def setup_llm_and_embeddings():
    max_mem = {0: "18GiB", 1: "20GiB"}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": 0},
        max_memory=max_mem,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        local_files_only=True,
        attn_implementation="sdpa",
    )

    # بهینه‌سازی پارامترها برای خروج از حالت تکرار سوال
    text_gen = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.3,  # افزایش ملایم برای روان‌تر شدن متن
        repetition_penalty=1.1,  # کاهش برای جلوگیری از کوتاه شدن بیش از حد پاسخ
        do_sample=True,
        top_p=0.9,
        top_k=50,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_PATH,
        model_kwargs={"device": "cuda:1"},
        encode_kwargs={"normalize_embeddings": True},
    )

    rerank_tokenizer = AutoTokenizer.from_pretrained(
        RERANKER_PATH, local_files_only=True
    )
    rerank_model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_PATH, local_files_only=True
    )
    rerank_model.to("cuda:1").eval()

    return (
        embeddings,
        HuggingFacePipeline(pipeline=text_gen),
        PROMPT_TEMPLATE,
        (rerank_model, rerank_tokenizer),
    )
