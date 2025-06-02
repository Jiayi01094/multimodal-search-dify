import pandas as pd
import torch
import json
import faiss
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# ---------- CONFIG ----------
CSV_PATH = "marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv"
FAISS_INDEX_PATH = "product_index.faiss"
METADATA_PATH = "product_metadata.json"
# ----------------------------

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load CSV
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")

# Metadata builder
def generate_structured_metadata(row):
    title = row.get("Product Name", "未知标题")
    brand = row.get("Brand Name") or "未知品牌"
    image_url = row.get("Image") or ""

    feature_fields = ["Product Description", "Product Specification", "About Product"]
    features = [str(row[col]).strip() for col in feature_fields if pd.notna(row.get(col)) and str(row[col]).strip()]
    features = [f for f in features if f.lower() != "nan"]

    usage = " | ".join([cat.strip() for cat in str(row.get("Category", "")).split("|") if cat.strip()])
    if not usage:
        usage = "通用场景"

    return {
        "title": title,
        "brand": brand,
        "image_url": image_url,
        "features": features,
        "usage": usage
    }

# Embedding + Metadata
embeddings = []
metadata = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        meta = generate_structured_metadata(row)
        image_urls = meta["image_url"].split("|")
        main_url = next((u for u in image_urls if u.strip().endswith((".jpg", ".png"))), None)
        if not main_url:
            continue

        response = requests.get(main_url.strip(), timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy())

        meta["image_url"] = main_url.strip()
        metadata.append(meta)

    except Exception as e:
        print(f"❌ Row {i} failed: {e}")

# Save FAISS index
emb_matrix = torch.vstack([torch.tensor(e) for e in embeddings]).numpy().astype("float32")
index = faiss.IndexFlatL2(emb_matrix.shape[1])
index.add(emb_matrix)
faiss.write_index(index, FAISS_INDEX_PATH)
print(f"✅ Saved FAISS index to {FAISS_INDEX_PATH}")

# Save metadata
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"✅ Saved metadata to {METADATA_PATH}")
