# 🧠 Multimodal Product Search API (CLIP + FAISS + FastAPI)

A multimodal conversational retrieval system that supports **image and text-based product search**. The backend is powered by [CLIP](https://openai.com/research/clip), [FAISS](https://github.com/facebookresearch/faiss), and [FastAPI](https://fastapi.tiangolo.com/), and can be integrated with [Dify](https://docs.dify.ai/) as a custom tool for intelligent product recommendations and Q&A.

---

## 🔍 Features

- ✅ Text-to-product and Image-to-product similarity search
- ✅ Fast retrieval with FAISS vector index
- ✅ OpenAPI-compatible interface for Dify or other LLM workflows
- ✅ Easily extensible and Docker-deployable

---

## 🧱 Project Structure

multimodal-search-dify/
│
├── backend/ # FastAPI API service
│ ├── main.py # API entry point (/search)
│ ├── preprocessor2.py # FAISS index loading/search
│ ├── product_metadata.json # Metadata of all products
│ ├── product_index.faiss # FAISS index file
│ ├── requirements.txt # Python dependencies
│ └── Dockerfile # For Docker packaging
│
├── dify_tool_schema.json # Tool schema to import into Dify
├── README.md

yaml

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/multimodal-search-dify.git
cd multimodal-search-dify/backend
2. Build and Run with Docker
docker build -t multimodal-api .
docker run -p 8000:8000 multimodal-api
The API will be live at: http://localhost:8000/docs

🛠 API Usage
Endpoint: POST /search
Request Parameters (multipart/form-data)
Field	Type	Required	Description
query	string	optional	Text query
image	file	optional	Product image
top_k	integer	optional	Number of results to return

Example Response:
json
{
  "results": [
    {
      "title": "Wireless Bluetooth Earbuds",
      "brand": "SoundX",
      "price": "$49.99",
      "features": ["Noise Cancellation", "Touch Controls"],
      "usage": "Ideal for workouts and travel"
    }
  ]
}
🤖 Integrate with Dify
In your Dify Flow, add a Tool and import dify_tool_schema.json.

Point the tool to: http://<your-server-ip>:8000/search

Use this tool in your LLM flow to query products via text or images.

📦 Add New Products (Optional)
To add new products:

Add metadata to product_metadata.json

Encode with CLIP and update product_index.faiss

(Or use an incremental script like update_index.py)

📚 Dependencies
faiss-cpu

transformers

torch

fastapi

uvicorn

Pillow

Install locally:

bash
pip install -r requirements.txt



MIT License © 2025 Jiayi Wang

