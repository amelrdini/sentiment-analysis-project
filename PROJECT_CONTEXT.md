# PROJECT CONTEXT — Indonesian Hate Speech Analyzer (Kelompok 5)

> **Cara pakai:** Paste seluruh isi file ini di awal percakapan dengan AI agent apa pun (OpenCode, Claude, ChatGPT, dst) sebelum minta bantuan lanjutan. Ini berisi seluruh konteks, keputusan, dan status proyek sampai sejauh ini — supaya agent tidak perlu re-explore dari nol.

---

## 1. Identitas Proyek

- **Mata kuliah:** Workshop Proyek Sistem Cerdas
- **Kelompok:** Kelompok 5 dari 6 kelompok
- **Tugas produk:** Indonesian Hate Speech Analyzer
- **Arsitektur wajib kelompok kami:** IndoBERT dan IndoBERTweet (kategori Transformer/Attention Mechanism)
- **Fokus kontribusi jurnal:** Analisis domain spesifik media sosial — membandingkan performa IndoBERT (general-domain) vs IndoBERTweet (domain-adaptive) pada deteksi hate speech Indonesia
- **Dosen:** Bu Selvia Ferdiana Kusuma, M.Kom

## 2. Requirement Wajib dari Kontrak Kuliah (Tidak Bisa Diubah)

- Dataset bersama satu kelas: **indotoxic2024**
- Task utama: **klasifikasi biner** (toxicity vs non-toxic)
- Task lanjutan: **klasifikasi multi-label** (5 kategori: profanity_obscenity, threat_incitement_to_violence, insults, identity_attack, sexually_explicit)
- Metrik wajib: **Macro-F1, Precision, Recall, Confusion Matrix**
- Wajib train/validation/test split
- Wajib uji perbandingan penanganan imbalance: **Baseline vs Class-weight/Focal loss vs Augmentasi data**
- Syarat mutlak: **reprodusibel sepenuhnya** dari data mentah sampai hasil akhir
- Prototype akhir wajib punya **4 komponen output**: Prediction, Confidence, Severity & Target, Explanation
- Penilaian: 10% Partisipasi, 20% Tugas Mingguan, 20% UTS (demo pipeline, minggu 8), 20% Implementasi Teknis (kualitas kode, reprodusibilitas, commit history), 30% UAS (presentasi final + laporan jurnal, minggu 15-16)
- **PJMK ke dosen (15/9):** materi PPT progres berdasarkan EDA (yang sudah diperbaiki) + preprocessing dari diskusi kelompok masing-masing. Dosen juga menanyakan strategi eksekusi eksperimen (data dipakai sekaligus/bertahap) — jawaban kelompok: bertahap (smoke test dulu, baru full run).

## 3. Struktur & Karakteristik Dataset (indotoxic2024)

### File
- `indotoxic2024_annotated_data-3.jsonl` — 43.692 baris (~34,6 MB)
- `indotoxic2024_annotator_data.jsonl` — 19 baris (profil demografis anotator)

### Struktur Kritis
**1 baris = 1 anotasi, BUKAN 1 baris = 1 teks.** Dari 43.692 anotasi, ada **28.449 teks unik** (`text_id`), dengan jumlah anotator per teks bervariasi 1–13.

### Nama Kolom Label ASLI (jangan pernah diganti/disalahartikan)
```
toxicity                          # label utama, biner
profanity_obscenity
threat_incitement_to_violence
insults
identity_attack
sexually_explicit
```
⚠️ **Bukan** `toxic/severe_toxic/obscene/threat/insult/identity_hate` (itu skema dataset Jigsaw bahasa Inggris, PERNAH tertukar di sesi awal — sudah dikoreksi).

### Kolom Lain yang Relevan
- `is_noise_or_spam_text` (0/1) — flag noise/spam bawaan dataset, **belum dipastikan apakah sudah difilter di pipeline saat ini** (sedang dicek)
- `related_to_election_2024` (0/1)
- `topic` — multi-label comma-separated (Disabilitas, Jewish, Kristen, LGBTQ+, Rohingya, Syiah, Ahmadiyah, Terpolarisasi, Tionghoa; ada anomali "1" dan "UNKNOWN")

### Profil Anotator
19 orang (12 perempuan, 7 laki-laki), usia 19–50, 11 etnis, 8 agama — didesain sengaja beragam.

## 4. Keputusan Metodologis yang Sudah Diambil

### 4.1 Agregasi Label (Step 2-3)
- **Metode:** Majority voting murni (`v/n >= 0.5`)
- **Tie-breaking:** Safety-first — semua tie diresolve ke **POSITIF**, berlaku konsisten untuk **SEMUA 6 label** (bukan selektif)
- **Justifikasi:** false negative lebih berbahaya daripada false positive untuk hate speech
- **Metadata disimpan:** skor agreement per teks per label (`agreement_<label>` = proporsi vote positif), flag `tie_break_applied` — untuk keperluan error analysis dan ablation study nanti (bukan untuk training langsung)

### 4.2 Pembersihan Data (Step 2-3)
- **Konten bervarian per `text_id`:** diambil modus (canonical text), baris minoritas dibuang (~171 baris)
- **Teks kosong:** dipulihkan dari anotasi lain dengan `text_id` sama jika tersedia (12 dari 13 kasus berhasil dipulihkan, hanya 1 teks — `2-1256` — benar-benar dibuang karena tidak ada teks valid sama sekali)
- **Duplikasi konten lintas `text_id`:** ditemukan 15,5% konten menyebar ke beberapa `text_id` berbeda — **risiko data leakage** kalau split random biasa

### 4.3 Split Data (Step 3) — **SUDAH DIVERIFIKASI ZERO LEAKAGE**
- **Metode:** Grouping first → Multi-label Iterative Stratification pada level Group ID (bukan level sampel individual), mengacu pendekatan Sechidis et al. (2011)
- **Proses:** (1) hashing & grouping konten identik/near-duplicate ke `group_id`, (2) vektor label per grup = OR/max dari anggota grup, (3) iterative stratified split di level grup, (4) mapping balik seluruh anggota grup ke split yang sama
- **Seed:** 42
- **Hasil final (setelah restart total dari resumedata.md):**
  - Total unit teks final: **28.448**
  - Total grup final: **26.157**
  - Split: **train 19.986 / val 4.229 / test 4.233**
  - **Verifikasi independen (`demo/cek_split_leakage.py`):** 0 grup yang muncul di lebih dari satu split — dikonfirmasi dari file di disk langsung, bukan klaim notebook
- **Distribusi label per split (train vs val):** sebagian besar label seimbang rapi, kecuali `sexually_explicit` (train 0,61% vs val 0,38%) dan `profanity_obscenity` (train 2,72% vs val ~4,26%) — sedikit timpang akibat trade-off constraint grup vs leakage. **Trade-off yang diambil: lebih baik proporsi sedikit timpang daripada leakage.**

⚠️ **CATATAN PENTING:** Angka split lama (`train 20.882 / val 2.610 / test 2.611`, total 26.103) adalah dari **pipeline SEBELUM restart total** dan **SUDAH TIDAK BERLAKU**. Angka yang valid adalah yang di atas (19.986/4.229/4.233).

### 4.4 Quality Check `sexually_explicit` (kategori paling langka & bermasalah)
- Total 142 sampel (0,5% dari data)
- 125 dari 142 hanya dianotasi 2 orang (rawan tie), 15 oleh 1 orang
- **~17% ternyata iklan komersial** (jasa gigolo/pijat/mainan dewasa) — konten seksual tapi BUKAN hate speech
- Ditemukan mislabel (teks tanpa konten seksual sama sekali ikut tertandai)
- 2 duplikat persis + 13 near-duplikat
- **Implikasi wajib dicatat di laporan:** performa model untuk kategori ini harus diekspektasikan rendah dan noisy — ini masalah kualitas label, bukan cuma soal kuantitas data

### 4.5 Preprocessing untuk Model (Step 6 — `src/preprocessing.py`)
Keputusan desain (berbasis temuan EDA, bukan template generik):
- **Lowercase diterapkan** ke `text_clean` untuk konsistensi tokenizer
- **TAPI sinyal huruf kapital & huruf berulang TIDAK dibuang** — disimpan sebagai kolom fitur terpisah (`caps_ratio`, `n_elongated`) karena keduanya terbukti berkorelasi dengan toxicity dari EDA
- URL dan mention (@username) dihapus; RT dihapus
- Hashtag: teks di-preserve (tanpa simbol #), karena EDA menemukan hashtag lebih banyak di teks non-toxic (26,8% vs 20,3%) — jadi informatif, bukan noise
- **Normalisasi slang** pakai kamus alay dari Ibrohim & Budi (2019) — 15.166 entri bersih
  - **Sitasi resmi (terverifikasi dari ACL Anthology):** Muhammad Okky Ibrohim and Indra Budi. 2019. *Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter.* In Proceedings of the Third Workshop on Abusive Language Online (ALW3), pages 46–57, Florence, Italy. ACL. Link: https://aclanthology.org/W19-3506/ — DOI: https://doi.org/10.18653/v1/W19-3506
  - File disimpan permanen di `data/external/new_kamusalay.csv` (bukan download ulang tiap run)
- **Tidak ada stemming/stopword removal agresif** — tokenizer BERT butuh teks natural
- **Exception khusus:** `ht` di konteks medsos dipetakan ke "hashtag", bukan "hati" (perbaikan dari bug kamus)
- **Kata "kaya" SENGAJA TIDAK diubah** — ambigu inheren (bisa "kayak"/"seperti" atau "kaya"/rich), whitelist berisiko merusak ribuan kasus lain
- **Kolom `needs_manual_review`** ditambahkan (True jika `n_slang_replaced >= 50`) — 13 teks total (train 5, val 4, test 4) — untuk flag audit manual, bukan otomatis diperbaiki
- File preprocessed: `data/processed/{train,val,test}_preprocessed.jsonl` — memakai kolom `text_clean` untuk tokenisasi

### 4.6 Ekstraksi Fitur untuk Model Utama
**Tidak diperlukan ekstraksi fitur manual** (TF-IDF, bag-of-words, dst) untuk task klasifikasi utama — model Transformer (IndoBERT/IndoBERTweet) melakukan ekstraksi fitur otomatis lewat embedding/self-attention. Fitur struktural (`caps_ratio`, `n_elongated`, `n_hashtag`, `n_emoji`, `n_slang_replaced`) yang sudah ada **disimpan untuk dipakai nanti** di modul severity scoring dan error analysis — bukan sebagai input model klasifikasi utama.

## 5. Temuan EDA Kunci (Step 4 & 5 — untuk bahan PPT/laporan jurnal)

### Step 4 — EDA Konten
1. Ketimpangan label ~25:1 — `toxicity` 15,34% vs `sexually_explicit` 0,61% (data train) → wajib macro-F1 + class-weight/focal loss, bukan akurasi
2. 14,9% teks multi-label (≥2 label positif sekaligus). Korelasi terkuat: `toxicity`+`insults` (r=0,65), `toxicity`+`identity_attack` (r=0,60) → output model harus 6 sigmoid terpisah, bukan softmax
3. Median panjang teks 20 kata, p95=139, p99=302 → rekomendasi `max_length=256` token. Menariknya, teks toxic (hasil agregasi final) justru lebih panjang (42,4 kata) dari non-toxic (36,4 kata) — kebalikan pola di data mentah, karena tie-break safety-first mengangkat teks ambigu (yang cenderung lebih panjang) jadi toxic
4. Kosakata toxic didominasi **kata identitas kelompok** ("zionis israel", "umat islam", "jalur gaza"), bukan kata kasar murni → justifikasi kuat untuk pendekatan Transformer (paham konteks) dibanding filter kata sederhana
5. Fitur medsos signifikan: huruf berulang 20,3% (toxic) vs 13,6% (non-toxic); huruf kapital 43,1% vs 36,1%; hashtag justru lebih tinggi di non-toxic (26,8% vs 20,3%) → dasar hipotesis IndoBERTweet > IndoBERT
6. Topik sangat mempengaruhi rasio toxic: **Syiah 62,0%** dan **Terpolarisasi 46,7%** (atau versi angka lain: 40,6%/38,1%, tergantung notebook run) tertinggi; **Tionghoa 7,4%** dan **UNKNOWN 6,8%** terendah
7. Ditemukan kasus kualitatif menarik: teks berkata kasar tapi TIDAK toxic (ekspresi emosi), dan teks toxic TANPA kata kasar sama sekali (serangan lewat konteks) — bukti kuat filter kata saja tidak cukup

### Step 5 — EDA Bias Anotator
1. Beban kerja antar anotator sangat timpang (min 100, median ~2.449, maks ~4.679 teks)
2. Ada anotator "strict" (7 orang) dan "lenient" (9 orang) berdasarkan deviasi rate toxicity dari rata-rata global — variasi ini wajar dan didokumentasikan, bukan disembunyikan
3. Pada 5.235 teks yang dinilai kedua kelompok (dengan kontrol): gap terbesar di `identity_attack` — mayoritas (Islam) menandai 15,6% vs minoritas 8,7% (gap 6,9pp); `insults` sebaliknya (minoritas 13,0% vs mayoritas 10,8%)
4. **Agreement antar anotator tergolong lemah**: pairwise `toxicity` 74,4%, Cohen's kappa 0,369 (lemah-moderat)
5. **Krippendorff alpha `threat_incitement_to_violence` NEGATIF (-0,131)** — penilaian threat antar anotator secara statistik lebih acak dari kebetulan. Temuan penting untuk limitasi laporan: label ini sangat bergantung persepsi individu anotator
6. Agreement berbeda per topik: Terpolarisasi paling tinggi (~40%), topik SARA rata-rata ~21,5%, topik lain hanya ~6,8% — makin sensitif topiknya, makin sering perlu tie-break

## 6. Struktur Folder Proyek (Final, Sudah Diimplementasikan)

```
project/
├── Dataset/                  # dataset mentah ASLI — tidak pernah diubah, hash-verified
├── data/
│   ├── raw/                  # salinan 1:1 dari Dataset/, SHA-256 verified
│   ├── interim/              # agreement_metadata.jsonl, splits_summary.json
│   ├── external/             # new_kamusalay.csv (kamus normalisasi slang)
│   └── processed/            # train/val/test.jsonl (arsip hasil split) +
│                              # train/val/test_preprocessed.jsonl (dipakai untuk training)
├── notebooks/                # nama file mengikuti proses aktual, bukan template kaku
│   ├── step1_identifikasi_dataset.ipynb
│   ├── step2_analisis_agregasi.ipynb
│   ├── step3_pipeline_final.ipynb       # agregasi + cleaning + grouping + split
│   ├── step4_eda.ipynb                  # EDA konten
│   ├── step5_eda_annotator_bias.ipynb   # EDA bias anotator
│   └── step6_eda_post_preprocessing.ipynb
├── src/
│   ├── quality_check.py       # sudah ada
│   ├── flag_needs_review.py   # sudah ada
│   ├── preprocessing.py       # sudah ada, dengan HT exception
│   ├── dataset.py             # BELUM — akan dibangun (Step 7)
│   ├── train.py               # BELUM — akan dibangun (Step 7)
│   ├── evaluate.py            # BELUM
│   ├── explainability.py      # BELUM
│   └── severity_scoring.py    # BELUM
├── models/                    # kosong, untuk checkpoint fine-tuning nanti
├── app/                       # kosong, untuk Streamlit prototype nanti
├── reports/
│   ├── figures/                # 20 figur EDA siap-PPT
│   ├── 01_dataset_identification.md
│   ├── 06_preprocessing_decisions.md
│   ├── 06_anomali_top_slang_dump.md
│   └── (nomor lain menyusul: split_report, smoke_test, experiment_results, error_analysis)
├── demo/                      # script verifikasi/arsip (cek_split_leakage.py, dst)
├── config.yaml                # seed, hyperparameter, path — single source of truth
├── requirements.txt
├── .gitignore                 # exclude Dataset/, data/raw, data/processed, models/, cache
└── README.md                  # struktur folder + cara menjalankan + sumber kamus
```

## 7. Aturan & Batasan Kerja (WAJIB Dipatuhi AI Agent Manapun)

1. **JANGAN PERNAH menyentuh Git/GitHub** — tidak `git init/add/commit/push`, tidak membuat/mengubah `.git/`, `.gitignore` boleh dibuat isinya tapi tidak dieksekusi via git. Commit dilakukan manual oleh anggota kelompok agar history mencerminkan kontribusi individu (bagian penilaian).
2. **Jangan memanipulasi data/hasil evaluasi** apa pun alasannya (syarat integritas akademik kontrak kuliah).
3. **Laporkan keterbatasan secara jujur** (terutama soal `sexually_explicit` dan `threat_incitement_to_violence` yang kualitas labelnya bermasalah) — jangan disembunyikan demi angka terlihat bagus.
4. **Tunjukkan bukti mentah (kode + output aktual), bukan cuma ringkasan naratif**, terutama untuk klaim statistik/angka penting — riwayat proyek ini pernah beberapa kali dapat laporan yang keliru/tidak sesuai data asli sebelum diverifikasi ulang.
5. **File besar (Dataset/, data/raw, data/processed, models/) tidak masuk git** — dibagikan lewat Google Drive, link akan ditambahkan manual ke README oleh kelompok.
6. **Data besar disalin (bukan dipindah)** dari `Dataset/` ke `data/raw/` — `Dataset/` asli tidak boleh disentuh sama sekali.
7. Kalau ada ambiguitas krusial (path, nama kolom, library tidak tersedia) — **tanya dulu, jangan berasumsi sendiri.**

## 8. Status Saat Ini & Langkah Selanjutnya

**Selesai:** Step 1 (identifikasi dataset) → Step 2 (analisis skema agregasi) → Step 3 (eksekusi pipeline: agregasi + cleaning + grouping + split, zero-leakage terverifikasi) → Step 4 & 5 (EDA konten & bias anotator) → Step 6 (preprocessing + EDA pasca-preprocessing, verifikasi manual distorsi slang).

**Sedang berjalan:** 
- Cek kolom `is_noise_or_spam_text` — apakah sudah difilter di pipeline atau belum (belum ada jawaban final)

**Belum dikerjakan (urutan berikutnya):**
1. Verifikasi GPU/environment
2. Bangun `src/dataset.py` (tokenisasi, `id2label`/`label2id` eksplisit sesuai nama kolom asli, collate function biner + multi-label)
3. Bangun `src/train.py` (baca semua parameter dari `config.yaml`)
4. Smoke test 4 run (2 model × 2 task) dengan subset kecil — verifikasi shape, loss, mapping label, estimasi waktu
5. Training penuh 12 run (2 model × 2 task × 3 strategi imbalance) — **HANYA setelah smoke test dikonfirmasi aman**
6. Evaluasi lengkap (Macro-F1, Precision, Recall, Confusion Matrix) + error analysis (termasuk korelasi dengan skor agreement anotator rendah)
7. Pemilihan model terbaik
8. Bangun modul severity scoring & explainability (attention-based)
9. Bangun prototype Streamlit (4 komponen output wajib)
10. Finalisasi README & laporan jurnal

---

*Dokumen ini adalah snapshot konteks proyek. Update bagian "Status Saat Ini" setiap kali ada progres besar, supaya tetap jadi referensi akurat untuk sesi AI agent berikutnya.*