# StackMatch – Smart Question Recommendation for Developers

## Struktur Folder

```
StackMatch/
├── main.py
├── requirements.txt
├── README.md
├── model_artifacts/         ← taruh hasil download dari Colab di sini
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── faiss_index.faiss
│   └── dataframe.parquet
└── frontend/
    └── index.html
```

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Taruh model artifacts
Salin isi folder `model_artifacts/` hasil download dari Google Colab ke folder ini.

### 3. Jalankan server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Buka browser
```
http://localhost:8000
```

## API Endpoints

| Method | Endpoint        | Fungsi                              |
|--------|-----------------|-------------------------------------|
| GET    | /api/tags       | List 20 bahasa pemrograman + count  |
| GET    | /api/stats      | Statistik dataset                   |
| POST   | /api/search     | Cari jawaban (query + tag + top_k)  |
| POST   | /api/translate  | Terjemahkan teks ke bahasa lain     |
| GET    | /api/random     | Pertanyaan acak (opsional filter tag)|

## Fitur Frontend

- 🔍 Hybrid Search (TF-IDF + SBERT + AnswerScore)
- 🏷 Filter by 20 programming language tags
- 📊 Score visualization per hasil (Fusion, TF-IDF, SBERT)
- 🌐 Translate hasil ke bahasa manapun
- 📋 Copy jawaban ke clipboard
- 🔄 Question Explorer (random questions)
- 🕐 Search History (tersimpan di browser)
- 📱 Responsive semua device
- ⌨ Ctrl+Enter untuk search cepat
