import faiss
import torch
import os
from langchain_community.vectorstores import FAISS

def load_vectorstore_on_gpu(path, embeddings, device_id=1):
    if not os.path.exists(os.path.join(path, "index.faiss")):
        return None
    try:
        # بارگذاری روی CPU و سپس انتقال به GPU 1
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        res = faiss.StandardGpuResources()
        cpu_index = vectorstore.index
        gpu_index = faiss.index_cpu_to_gpu(res, device_id, cpu_index)
        vectorstore.index = gpu_index
        return vectorstore
    except Exception as e:
        print(f"FAISS GPU Error: {e}")
        return None