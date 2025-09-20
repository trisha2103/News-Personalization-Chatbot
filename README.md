# News Personalization — Vector Search (RAG) + Ranking + Daily Brief + Chatbot

Local, CPU-friendly project that turns the **MIND-small** news dataset into:

- **Semantic search** over articles (embeddings + cosine similarity)
- **RAG answers** with citations (TF-IDF + MMR)
- **Personalized ranking** from user click history
- **Daily Brief** generator (sectioned markdown)
- **Chatbot UI** (Streamlit)

---

## ✨ Features

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dim) on `title + abstract`
- Retriever: NumPy dot product (FAISS optional)
- RAG: TF-IDF relevance + MMR diversity → short, cited answer
- Personalization: user profile = mean of clicked article vectors (train split)
- UI: Streamlit chat with profile switcher, α (query vs taste), Top-K

---


## 📦 Requirements

Tested on Python **3.11–3.13** (no GPU required).

`requirements.txt`
 - streamlit==1.38.0
 - sentence-transformers==3.0.1
 - scikit-learn==1.5.1
 - numpy==1.26.4
 - pandas==2.2.2
 - nltk==3.9.1

---

## 📥 Data

Use **MIND-small** (Microsoft News). Download the **train** and **dev** splits and unzip them to:
 - data/MINDsmall_train/
 - data/MINDsmall_dev/

**Each split contains:**
- `news.tsv` — article metadata (id, category, title, abstract, url)
- `behaviors.tsv` — impressions with `user_id`, timestamp, and clicked/unclicked items

> Please follow the MIND dataset license/terms.

---

## 🛠 Build Artifacts

Open `src/db.ipynb` and run cells in order:

1. **SQLite build → `data/news.db`**  
   - Creates tables: `news`, `impressions`, `impression_items`.

2. **Checks / uniques**  
   - Fix duplicate inserts if any (e.g., use split-aware impression IDs).

3. **Embeddings (safe batches)**  
   - Model: `sentence-transformers/all-MiniLM-L6-v2`  
   - Outputs in `artifacts/`:  
     - `news_meta.csv`  
     - `news_embs.dat`  
     - `index_info.json`  
     - *(optional)* `news.faiss` (FAISS index)

4. **RAG answerer**  
   - Retrieval → TF-IDF + MMR → concise summary with **URLs**.

5. **Personalization**  
   - Build user profile vectors from **train** clicks; optionally evaluate **Hit@10** on dev.

6. **Daily Brief (optional)**  
   - Generate sectioned Markdown (Business / Finance / Health / Entertainment).

---

## ⚙️ How It Works

1. **Ingest**  
   MIND-small → SQLite (`data/news.db`) + tidy pandas views.

2. **Embed**  
   Titles + abstracts with MiniLM (384-dim) → `artifacts/news_embs.dat` (+ `news_meta.csv`, `index_info.json`).

3. **Retrieve**  
   Cosine similarity via NumPy dot product over normalized embeddings.

4. **RAG**  
   TF-IDF rank + **MMR** redundancy control → stitched answer + **citations** (source URLs).

5. **Personalize**  
   `score = α · sim(query, doc) + (1 − α) · sim(user_profile, doc)`

6. **Daily Brief**  
   Preset section queries + recency filter → Markdown summary (Business / Finance / Health / Entertainment).

---

## 🙏 Acknowledgments

- **MIND dataset** — Microsoft News
- **Sentence-Transformers** — MiniLM and related embedding models

