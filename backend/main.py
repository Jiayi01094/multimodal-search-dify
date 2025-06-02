# save as app.py

from fastapi import FastAPI, UploadFile, File, Form
from typing import List, Optional
from pydantic import BaseModel
import faiss
import torch
import json
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import numpy as np

app = FastAPI()

# Load models and data
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
index = faiss.read_index("product_index.faiss")
with open("product_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Response model
class Product(BaseModel):
    title: str
    features: List[str]
    usage: str
    price: Optional[str] = None
    image_url: Optional[str] = None
    brand: Optional[str] = None  # ✅ 改成 Optional

def embed_image(image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

def sanitize_product(product: dict) -> dict:
    """确保所有字段合法，尤其是字符串字段非 NaN"""
    def safe(val, default=""):
        if val is None:
            return default
        if isinstance(val, float) and np.isnan(val):
            return default
        return val
    product["title"] = safe(product.get("title"), "无标题")
    product["features"] = product.get("features") or ["未知功能"]
    product["usage"] = safe(product.get("usage"), "通用场景")
    product["price"] = safe(product.get("price"), "暂无价格")
    product["brand"] = safe(product.get("brand"), None)
    product["image_url"] = safe(product.get("image_url"), "")
    return product

from transformers import CLIPTokenizer

tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer

def embed_text(query: str) -> np.ndarray:
    inputs = processor(text=query, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


@app.post("/search")
async def search_products(
    image: UploadFile = File(...),
    query: Optional[str] = Form(None),
    top_k: int = Form(1)
):
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        vector = embed_image(img)
        if query:
            # 若同时上传文本，可扩展联合嵌入（此处省略）
            pass
        D, I = index.search(vector, top_k)
        results = [sanitize_product(metadata[i]) for i in I[0] if i < len(metadata)]
        return {"results": results}
    except Exception as e:
        print("❌ ERROR (image route):", e)
        return {"results": []}

# --------- 路由 2：纯文本搜索 ----------
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 1

@app.post("/search_by_query")
async def search_by_query(req: QueryRequest):
    try:
        vector = embed_text(req.query)
        D, I = index.search(vector, req.top_k)
        results = [sanitize_product(metadata[i]) for i in I[0] if i < len(metadata)]
        return {"results": results}
    except Exception as e:
        print("❌ ERROR (text route):", e)
        return {"results": []}

#uvicorn query_only:app --host 0.0.0.0 --port 8000 --reload
#https://ab4a-104-153-230-40.ngrok-free.app