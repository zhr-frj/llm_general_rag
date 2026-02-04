import faiss
import os
import gc
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ۱. تابع هوشمند برای بارگذاری دیتابیس روی GPU 1 بدون تغییر در پارامترهای جستجو
def load_vectorstore_on_gpu(folder_path, embeddings):
    """
    فقط انتقال دیتابیس به GPU 1 و پاکسازی حافظه سربار.
    بدون دستکاری در تنظیمات دقت و تعداد نتایج (k).
    """
    if not os.path.exists(folder_path):
        print(f"❌ خطا: دیتابیس در مسیر {folder_path} یافت نشد.")
        return None

    print(f"🚀 در حال لود دیتابیس و بهینه‌سازی حافظه...")

    # الف) بارگذاری در CPU
    vectorstore = FAISS.load_local(
        folder_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    try:
        # ب) آماده‌سازی منابع GPU 1
        res = faiss.StandardGpuResources()

        # ج) انتقال ایندکس به GPU (سرعت بخشیدن به جستجو بدون تغییر در k)
        gpu_index = faiss.index_cpu_to_gpu(res, 1, vectorstore.index)
        vectorstore.index = gpu_index

        # د) پاکسازی حافظه رزرو شده و اضافی (Garbage Collection)
        # این کار کش محاسباتی شما را پاک نمی‌کند، فقط فضای خالی برای n کاربر می‌سازد
        gc.collect()
        torch.cuda.empty_cache()

        print("✅ دیتابیس به GPU 1 منتقل و حافظه سربار تخلیه شد.")
    except Exception as e:
        print(f"⚠️ انتقال به GPU انجام نشد، اما برنامه در حالت CPU پایدار است: {e}")

    return vectorstore

# ۲. تابع جستجوی منعطف (بدون مقدار اجباری k)
def search_documents(vectorstore, query, *kwargs):
    """
    این تابع هر مقداری که قبلاً برای k داشتید را می‌پذیرد 
    و تغییری در منطق فعلی برنامه شما ایجاد نمی‌کند.
    """
    return vectorstore.similarity_search(query, *kwargs)
