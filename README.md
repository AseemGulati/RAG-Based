# 🤖 Gemini RAG Chatbot with FAISS + HuggingFace Embeddings
<img width="1913" height="899" alt="image" src="https://github.com/user-attachments/assets/ac023244-62f3-41a8-83d1-4b6156fcd812" />
<img width="1919" height="904" alt="image" src="https://github.com/user-attachments/assets/d2f316b8-ae1b-40d7-a5fa-5034e3304058" />


This Streamlit app enables users to interact with a PDF document using **RAG (Retrieval Augmented Generation)** powered by **Google Gemini-2**, **HuggingFace Embeddings**, and **FAISS** for fast vector similarity search.

---

## 🔍 Features

- Upload and analyze PDF documents
- Extract and chunk text intelligently
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Store and retrieve relevant chunks with **FAISS**
- Generate smart responses using **Gemini-2.0-Flash**
- Simple and responsive **Streamlit** interface

---

## 🧠 Architecture

📄 PDF Upload
↓
📚 Text Chunking (LangChain)
↓
🧬 Embedding (HuggingFace)
↓
📦 Vector Store (FAISS)
↓
📤 Query Input
↓
🔍 Retrieve Relevant Chunks
↓
🎯 Gemini-2.0 Response Generation


---

### 1. 📦 Installation
 **Clone the repo**
 
### 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Set up environment variable

Create a .env file with your Gemini API key:
GOOGLE-API-KEY=your_google_api_key_here

### 5. 🚀 Run the App
streamlit run app.py

##📄 Sample Use Case
Upload a contract, research paper, or technical document, and ask:

"What is the main objective of this paper?"
"List all the key terms mentioned in the agreement."
"Summarize this document in 3 points."
