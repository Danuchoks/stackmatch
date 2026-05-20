```markdown
# StackMatch
**Smart Question Recommendation for Developers**

StackMatch adalah aplikasi rekomendasi jawaban berbasis AI yang membantu developer menemukan jawaban Stack Overflow paling relevan. Cukup pilih bahasa pemrograman, ketik pertanyaan dalam bahasa apapun, dan StackMatch akan menemukan jawaban terbaik menggunakan teknologi Hybrid Search (TF-IDF + SBERT).

---

## Fitur

- Hybrid Search: gabungan TF-IDF (40%) + SBERT Semantic (50%) + Vote Score (10%)
- Filter pencarian berdasarkan 20 bahasa pemrograman
- Input pertanyaan dalam bahasa apapun, termasuk Bahasa Indonesia
- Terjemahkan jawaban ke 12+ bahasa
- Visualisasi skor relevansi per hasil (Fusion, TF-IDF, SBERT)
- Riwayat pencarian tersimpan di browser
- Tampilan Dark dan Light mode
- Responsif di semua ukuran layar

---

## Cara Kerja

Query masuk → diterjemahkan ke English → diproses TF-IDF + SBERT → digabung dengan Fusion Score → hasil diurutkan berdasarkan relevansi.

---

## Bahasa Pemrograman yang Didukung

assembly, c#, c++, dart, go, haskell, java, javascript, kotlin, lua, objective-c, perl, php, python, r, ruby, rust, scala, swift, typescript

---

## Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/danugans/stackmatch.git
cd stackmatch
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Siapkan Model Artifacts
Taruh file berikut di dalam folder `model_artifacts/`:
```
model_artifacts/
├── tfidf_vectorizer.pkl
├── tfidf_matrix.pkl
├── faiss_index.faiss
└── dataframe.parquet
```

### 4. Jalankan Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Buka Browser
```
http://localhost:8000
```

---

## Cara Menggunakan

1. Pilih tag bahasa pemrograman yang sesuai dengan pertanyaanmu
2. Ketik pertanyaan, bisa dalam Bahasa Indonesia atau Inggris
3. Atur jumlah hasil yang diinginkan lewat slider
4. Klik Find Answers atau tekan Ctrl+Enter
5. Klik Translate pada hasil jika ingin diterjemahkan
6. Klik Copy untuk menyalin jawaban ke clipboard

---

## API Endpoints

| Method | Endpoint | Deskripsi |
|---|---|---|
| GET | `/api/tags` | List 20 bahasa dan jumlah data |
| GET | `/api/stats` | Statistik dataset |
| POST | `/api/search` | Cari jawaban |
| POST | `/api/translate` | Terjemahkan teks |
| GET | `/api/random` | Pertanyaan acak |

---

## Tech Stack

- Backend: FastAPI + Uvicorn
- Semantic Search: SBERT (all-MiniLM-L6-v2)
- Keyword Search: TF-IDF (scikit-learn)
- Vector Index: FAISS
- Translation: deep-translator
- Frontend: HTML, CSS, Bootstrap 5

---

## Struktur Folder

```
StackMatch/
├── main.py
├── requirements.txt
├── README.md
├── model_artifacts/
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── faiss_index.faiss
│   └── dataframe.parquet
└── frontend/
    └── index.html
```

---

Dibuat sebagai Capstone Project program SIB (Studi Independen Bersertifikat).
```
