# Rancangan Metode Penelitian — Deteksi Hate Speech Berbahasa Indonesia

**Proyek:** Indonesian Hate Speech Analyzer — Workshop Proyek Sistem Cerdas, Kelompok 5
**Judul kerja:** Perbandingan Fine-Tuning IndoBERT vs IndoBERTweet untuk Klasifikasi Multi-Label Hate Speech Berbahasa Indonesia (IndoToxic2024)
**Status dokumen:** angka tahap 1–6 adalah hasil aktual yang sudah dieksekusi dan terverifikasi; tahap 7–9 masih **direncanakan** (belum ada training/evaluasi yang selesai).

---

## 1. Data Asli

Sumber: `Dataset/indotoxic2024_annotated_data-3.jsonl` — publikasi Genta Indra Winata et al., 2024 (media sosial berbahasa Indonesia, konteks Pemilu 2024).

| Aspek | Nilai aktual |
|---|---|
| Baris mentah (anotasi) | **43.692** (34,6 MB, JSONL) |
| Profil anotator | `indotoxic2024_annotator_data.jsonl` — **19 anotator** (ID 1–21 tanpa 14 & 17; 12 P, 7 L; usia 20–50) |
| Teks unik (`text_id`) | **28.449** |
| Struktur | 1 baris = 1 anotasi (satu teks dinilai satu anotator), **bukan** 1 baris = 1 teks |
| Anotasi per teks | 1 anotasi: 18.039 teks · 2: 9.864 · 3: 96 · 5: 1 · 10: 5 · 11: 95 · 13: 349 |
| Kolom label | 6 label biner per anotasi: `toxicity`, `profanity_obscenity`, `threat_incitement_to_violence`, `insults`, `identity_attack`, `sexually_explicit` |
| Metadata tambahan | `topic` (multi-label dipisah koma), `is_noise_or_spam_text`, `related_to_election_2024`, `initial_paragraph` (terisi 4.210 baris) |

**Distribusi label level anotasi (43.692 baris, dari `reports/01_dataset_identification.md`):**

| Label | Positif | % |
|---|---|---|
| toxicity | 6.899 | 15,8% |
| profanity_obscenity | 1.294 | 3,0% |
| threat_incitement_to_violence | 1.504 | 3,4% |
| insults | 3.222 | 7,4% |
| identity_attack | 3.197 | 7,3% |
| sexually_explicit | 235 | 0,5% |
| is_noise_or_spam_text | 2.937 | 6,7% |
| related_to_election_2024 | 4.822 | 11,0% |

**Anomali yang sudah teridentifikasi sejak Step 1:**
- 2 baris memiliki `text` kosong/berisi spasi: baris fisik 1256 (`text_id 2-1256`, annotator 21) dan baris fisik 28394 (`text_id 103-150`, annotator 6); keduanya ber-`topic: "UNKNOWN"`.
- Nilai anomali `topic`: `"1"` dan `"UNKNOWN"`.

---

## 2. Tahapan Proses (yang sudah dieksekusi)

### 2.1 Pembersihan data (notebook `step3_pipeline_final.ipynb`, seed 42)

**Apa yang dilakukan:**
1. **Kanonikalisasi konten** — untuk setiap `text_id` dengan beberapa anotasi, teks kanonik dipilih sebagai **modus konten** (hasil `strip`). Baris yang kontennya berbeda dari kanonik dibuang.
2. **Pemulihan teks kosong** — unit yang seluruh anotasinya kosong dibuang; unit yang hanya *sebagian* anotasinya kosong tetap dipulihkan lewat teks kanonik.
3. **Normalisasi topik** — `topic "1"` digabung ke `"UNKNOWN"`; spasi koma diseragamkan (`", "`).

**Hasil kuantitatif (log aktual notebook + `splits_summary.json`):**

| Metrik | Nilai |
|---|---|
| Baris konten minoritas dibuang | 171 |
| Unit kosong dibuang | 1 (text_id `2-1256`; 1 baris) |
| Unit kosong terpulihkan | 1 (text_id `103-150`) |
| Baris valid setelah bersih | **43.520** |
| Teks unik setelah bersih | **28.448** |
| topic "1" tersisa | 0 |
| topic UNKNOWN | 3.903 |

### 2.2 Agregasi label multi-anotator

**Metode:** `majority_safety_first` — untuk tiap teks dan tiap label, label final = 1 jika `v/n >= 0,5` (v = jumlah vote positif, n = jumlah anotator). Konsekuensi tie pada n genap (`v = n/2`) adalah **label positif** — keputusan safety-first untuk **seluruh 6 label** (dokumentasi keputusan: `reports/step2_prompt.md`). Disimpan juga metadata `agreement_<label>` (rasio v/n) dan flag boolean `tie_break_applied_<label>` untuk ablasi mendatang.

Rekonsiliasi angka (sesi awal vs final): any-positive toxicity = 4.236, majority murni = 2.404, single-positive = 1.832; metode final memakai safety-first sehingga hasilnya 4.414.

**Hasil kuantitatif (28.448 unit):**

| Label | Positif | % | Tie-break |
|---|---|---|---|
| toxicity | 4.414 | 15,5% | 1.996 |
| profanity_obscenity | 904 | 3,2% | 519 |
| threat_incitement_to_violence | 1.068 | 3,8% | 945 |
| insults | 2.260 | 7,9% | 1.376 |
| identity_attack | 2.186 | 7,7% | 1.242 |
| sexually_explicit | 151 | 0,5% | 95 |

Distribusi `n_annotators` per unit: 1 → 18.169; 2 → 9.759; 3 → 70; 5 → 1; 10 → 7; 11 → 95; 12 → 8; 13 → 339.

### 2.3 Grouping & split data

**Grouping:** kunci grup = konten dinormalisasi (lowercase, whitespace berlebih & tanda kutip dihapus). Semua `text_id` berkonten identik/near-duplicate → satu `group_id`, dijamin masuk split yang sama.

**Metode split:** **Group-aware Multi-label Iterative Stratification** (`IterativeStratification` dari `scikit-multilearn` 0.2.0; `n_splits=3`, `order=1`, `sample_distribution_per_fold=[0.70, 0.15, 0.15]`). Stratifikasi dijalankan **di level grup** memakai vektor label agregat grup (OR/Max antar anggota), lalu hasilnya di-*re-map* ke semua unit anggota grup. Tujuan: mencegah leakage near-duplicate sambil menjaga proporsi 6 label di tiap split. Seed 42.

**Hasil kuantitatif:**

| Split | Unit (teks) | Grup |
|---|---|---|
| train (70%) | 19.986 | 18.309 |
| val (15%) | 4.229 | 3.924 |
| test (15%) | 4.233 | 3.924 |
| total | 28.448 | 26.157 |

- Unit dalam grup multi-anggota: 4.425 (15,6%).
- **Leakage check: 0 grup lintas split** (diverifikasi ulang di `demo/cek_angka_flag.py`: irisan grup train∩val = train∩test = val∩test = 0; total grup unik 26.157).

Distribusi label per split (unit-level, dari `splits_summary.json`): toxicity train 15,3% / val 16,0% / test 15,8%; profanity 2,7% / 4,3% / 4,3%; threat 3,7% / 3,8% / 3,8%; insults 7,8% / 8,4% / 8,4%; identity_attack 7,6% / 7,9% / 7,8%; sexually_explicit 0,6% / 0,4% / 0,3%.

**Output:** `data/processed/{train,val,test}.jsonl` (1 baris = 1 teks berlabel final) + `agreement_metadata.jsonl` + `splits_summary.json` (arsip di `data/interim/`).

### 2.4 EDA

Notebook `step4_eda.ipynb` (konten), `step5_eda_annotator_bias.ipynb` (19 anotator, agreement, bias demografi), `step6_eda_post_preprocessing.ipynb` (pasca-preprocessing) — menghasilkan **23 figur** di `reports/figures/` (01–13 EDA konten; 14–16 toxic vs non-toxic, agreement & bias demografi; 17–21 EDA pasca-preprocessing). Temuan kunci yang memengaruhi desain preprocessing: kapital berkorelasi dengan toxicity (43,1% vs 36,1%), huruf berulang (20,3% vs 13,6%), 21,4% teks memuat emoji.

### 2.5 Preprocessing teks (`src/preprocessing.py`)

Input `data/processed/{split}.jsonl` → output `{split}_preprocessed.jsonl` (train 19.986, val 4.229, test 4.233 baris). Per baris disimpan: `text` (asli, kapital utuh), `text_clean` (siap tokenizer), `fitur` (10 fitur struktural), `needs_manual_review`.

Urutan langkah pembentukan `text_clean` (persis di `src/preprocessing.py`):
1. **Fitur dihitung dari teks asli sebelum perubahan apa pun:** `caps_ratio`, `n_allcaps_word`, `n_elongated`, `n_url`, `n_mention`, `n_hashtag`, `n_emoji`, `n_word`, `n_char` (+ `n_slang_replaced` setelah penggantian).
2. **URL dihapus** seluruhnya (`https?://…|www.…`), jumlah dicatat `n_url` — tidak informatif (link mati/iklan).
3. **Mention `@user` dihapus** seluruhnya, jumlah dicatat `n_mention` — hanya identitas akun, bukan konten.
4. **Hashtag:** simbol `#` dibuang, **katanya dipertahankan** (informasi topik, mis. `#EastTurkistan`); jumlah dicatat `n_hashtag`.
5. **Emoji dihapus**, jumlah dicatat `n_emoji` — tidak dipahami tokenizer BERT.
6. **Kompresi huruf berulang** run ≥3 → 2 (`anjirrrr` → `anjirr`), jumlah kata terdampak dicatat `n_elongated`.
7. **Lowercase** — sinyal kapital sudah tersimpan sebagai fitur terpisah.
8. **Pengecualian `ht` → `hashtag`** sebelum kamus dipakai (mencegah mapping keliru `ht → hati`; setelah perbaikan frasa "hati cina" tersisa 1 di train, 0 di val/test).
9. **Normalisasi slang word-by-word** memakai kamus `new_kamusalay.csv` (15.166 entri; pencocokan kata penuh setelah strip tanda baca, bukan substring).
10. **Rapikan whitespace.**

**Tidak dilakukan:** stemming dan stopword removal — tokenizer WordPiece butuh teks natural berimbuhan ("menandai" vs "tandai" berbeda makna).

**Audit anomali kamus** (`reports/06_anomali_top_slang_dump.md`): 5 teks dengan penggantian slang tertinggi (111, 72, 60, 59, 52) diaudit manual — distorsi nyata terkonsentrasi di teks rant/katalog panjang dan daftar username; kelima teks berlabel `toxicity=0`. Mapping `kaya → kayak` sengaja tidak diberi pengecualian (ambigu inheren, dicatat sebagai keterbatasan).

**Flag `needs_manual_review`** = `n_slang_replaced >= 50` (tidak dipakai saat training, hanya untuk error analysis): **train 5, val 4, test 4 (total 13 teks)** — angka hasil eksekusi `src/flag_needs_review.py`, diverifikasi ulang via `demo/cek_angka_flag.py`.

### 2.6 Struktur data final untuk training

Kolom `train_preprocessed.jsonl` (aktual): `text_id`, `text`, `initial_paragraph`, `topic`, `n_annotators`, 6 label final + `agreement_*` + `tie_break_applied_*`, `is_noise_or_spam_text`, `related_to_election_2024`, `group_key`, `group_id`, `split`, `text_clean`, `fitur` (10 fitur), `needs_manual_review`.

---

## 3. Metode Klasifikasi (DIRENCANAKAN)

> Seluruh sub-bab ini adalah **rencana** — belum ada checkpoint model terlatih di `models/` (folder `models/indobert/` dan `models/indobertweet/` masih kosong).

### 3.1 Model

| Aspek | Model A (baseline general-domain) | Model B (domain-adaptive) |
|---|---|---|
| Checkpoint HF | `indobenchmark/indobert-base-p1` | `indolem/indobertweet-base-uncased` |
| Pre-training | 220 juta kata Wikipedia+ Indonesia (IndoLEM/IndoNLU) | Pre-training lanjutan IndoBERT pada 286 juta tweet Indonesia (Koto et al., IndoLEM) |
| Arsitektur | 12 layer, 768 hidden, 12 head, ~124 jt parameter | idem, ~124 jt parameter |
| Tokenizer | WordPiece 50.000 (uncased, p1) | WordPiece 31.923 (uncased, khas media sosial) |

### 3.2 Dua task paralel

- **Task biner (2 kelas):** klasifikasi `toxicity` saja → classification head dense + **2 unit output** (softmax/BCE), input sama.
- **Task multi-label (5 label):** 5 output sigmoid independen: `profanity_obscenity`, `threat_incitement_to_violence`, `insults`, `identity_attack`, `sexually_explicit` — bukan softmax; loss `BCEWithLogitsLoss` per label.

> Catatan desain: artefak pipeline yang sudah dieksekusi (config.yaml, `LABEL_COLS` di step 3 dan preprocessing) menyimpan vektor **6 label termasuk toxicity** sebagai satu kesatuan data — itu kenapa bagan awal (figur 22) memakai head 6-sigmoid. Pemisahan menjadi 2 task (biner toxicity + multi-label 5 label) adalah keputusan desain modeling yang diambil setelahnya; datanya tidak berubah, hanya cara memotong target saat training: task biner memakai kolom `toxicity`, task multi-label memakai 5 kolom sisanya.

### 3.3 Library & versi (dari `requirements.txt` aktual)

| Library | Versi | Peran |
|---|---|---|
| numpy | 1.26.4 | komputasi numerik |
| pandas | 2.1.4 | manipulasi data |
| matplotlib | 3.7.5 | visualisasi |
| seaborn | 0.13.2 | visualisasi |
| wordcloud | 1.9.6 | wordcloud EDA |
| scikit-learn | 1.4.2 | semua metrik evaluasi |
| scikit-multilearn | 0.2.0 | IterativeStratification (split) |
| jupyter | (latest) | notebook |
| torch | (belum di-pin) | training & inference |
| transformers | (belum di-pin) | AutoTokenizer + AutoModelForSequenceClassification |

Catatan: torch dan transformers belum dipin versinya di `requirements.txt` — akan dipin saat modul training final dibuat. Python 3.11 (kernel notebook).

### 3.4 Hyperparameter (rencana; config.yaml belum memuat blok training)

`config.yaml` saat ini hanya memuat: `seed: 42`, `labels` (6 label), `split` (0.70/0.15/0.15), `aggregation` (majority_safety_first). Nilai training berikut mengacu pada bagan yang sudah disepakati (`demo/bagan_penelitian.py` / `reports/figures/22_bagan_penelitian.png`) dan akan difinalisasi di `config.yaml` + `src/train.py`:

| Parameter | Nilai rencana |
|---|---|
| input | `text_clean`, `max_length = 128` token, padding + truncation |
| classification head | dropout + dense → 5 unit (multi-label, sigmoid) / 2 unit (biner) |
| loss | `BCEWithLogitsLoss` (multi-label); BCE/CE (biner) |
| optimizer | AdamW, `lr = 2e-5`, weight decay 0,01 |
| batch size | 32 |
| epoch | maks 4, early stopping berdasarkan val macro-F1 |
| seed | 42 |
| checkpoint | `models/indobert/`, `models/indobertweet/` |

### 3.5 Strategi penanganan imbalance (rencana, dibandingkan sebagai eksperimen)

Distribusi label sangat timpang (sexually_explicit 0,5%; profanity 3,2%). Tiga strategi yang akan dibandingkan dengan protokol dan test set yang sama:
1. **Baseline** — BCE tanpa pembobotan (referensi).
2. **Class-weight / focal loss** — bobot kelas invers-proporsional atau focal loss untuk menekan dominasi kelas negatif.
3. **Augmentasi data** — untuk kelas langka (mis. back-translation / EDA), direncanakan sebagai tahap lanjutan.

Metrik pembanding utama: **macro-F1 test** (agar label langka setara bobotnya dengan label mayoritas), didukung PR-AUC per label.

---

## 4. Kamus / Lexicon yang Dipakai (sudah dipakai di preprocessing)

- **Nama:** `new_kamusalay.csv` — kamus normalisasi slang/alay Indonesia (format csv dua kolom `alay,normal`).
- **Lokasi:** `data/external/new_kamusalay.csv` (disimpan permanen untuk reproducibility, tidak diunduh ulang tiap run).
- **Jumlah entri bersih:** **15.166** (setelah lompat baris kosong + buang duplikat; kunci di-lowercase).
- **Pemakaian:** penggantian kata-dua-kata (word-by-word) di `text_clean`, bukan substring match, agar kata lain yang kebetulan memuat substring tidak ikut terganti. Contoh: "kemaren"→"kemarin", "bgt"→"banget", "praktek"→"praktik".
- **Sitasi lengkap:**
  Muhammad Okky Ibrohim and Indra Budi. 2019. Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter. In *Proceedings of the Third Workshop on Abusive Language Online (ALW3)*, pages 46–57, Florence, Italy. Association for Computational Linguistics.
- **Link:** https://aclanthology.org/W19-3506/ · **DOI:** https://doi.org/10.18653/v1/W19-3506
- **Sumber file:** https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv

---

## 5. Matriks Evaluasi (DIRENCANAKAN)

### 5.1 Metrik dan definisi

Semua dihitung dengan **scikit-learn 1.4.2** (`sklearn.metrics`), pada test set hold-out (4.233 teks), threshold sigmoid default 0,5.

| Metrik | Definisi/rumus | Fungsi sklearn |
|---|---|---|
| Precision (per label) | TP / (TP + FP) | `classification_report`, `precision_recall_fscore_support` |
| Recall (per label) | TP / (TP + FN) | idem |
| F1-score (per label) | 2·P·R / (P + R) | idem |
| Macro average | rata-rata metrik per label tanpa bobot → **metrik utama pemilihan model** (label langka setara) | `average="macro"` |
| Micro average | agregat TP/FP/FN lintas label → didominasi label mayoritas | `average="micro"` |
| Weighted average | rata-rata per label dibobot support | `average="weighted"` |
| Subset accuracy (exact match) | proporsi sampel dengan **seluruh 5 label benar** (task multi-label) / 1 label (task biner) | `accuracy_score` pada vektor multi-label |
| Hamming loss | rata-rata proporsi label salah per sampel | `hamming_loss` |
| ROC-AUC per label | luas kurva TPR vs FPR saat threshold digeser | `roc_auc_score` |
| PR-AUC per label | luas kurva precision vs recall — lebih informatif untuk label langka (sexually_explicit 0,5%) | `average_precision_score` |
| Confusion matrix | TP/FP/FN/TN per label (one-vs-rest) | `multilabel_confusion_matrix` (6 matriks 2×2) |

### 5.2 Membaca confusion matrix untuk task multi-label

`multilabel_confusion_matrix` menghasilkan **satu confusion matrix 2×2 per label** (5 matriks untuk task multi-label, 1 matriks untuk task biner), bukan satu matriks 5×5: setiap label dibaca sebagai masalah biner one-vs-rest (baris = label aktual, kolom = prediksi). Interpretasi dilakukan **per label** (mis. matrix `toxicity` membaca kinerja deteksi toxicity semata) — karena satu teks bisa bersamaan benar di satu label dan salah di label lain, tidak ada "satu matriks agregat" yang bermakna; agregasi dilakukan lewat macro/micro average di atas. FN per label langka (sexually_explicit) menjadi fokus error analysis.

### 5.3 Protokol perbandingan (model & strategi imbalance)

1. Evaluasi **hanya di test set 4.233 teks** (zero-leakage grup, tidak pernah disentuh training/tuning).
2. `classification_report` per label + macro/micro/weighted untuk tiap model (IndoBERT vs IndoBERTweet) dan tiap strategi imbalance (baseline vs class-weight/focal vs augmentasi) — semua diuji pada test set yang sama, seed sama.
3. Perbandingan utama antar konfigurasi: **macro-F1**; label langka dibaca lewat PR-AUC & recall per label.
4. Uji signifikansi: **bootstrap 1.000 resample (CI 95% macro-F1)** + **McNemar test** per label.
5. Error analysis kualitatif: FP/FN terbanyak + 13 teks `needs_manual_review` diisolasi untuk memeriksa apakah kesalahan model berasal dari label salah atau distorsi preprocessing.
6. Model terbaik → prototype `app/` Streamlit (inference teks bebas).

---

## Lampiran: Jejak artefak aktual yang dikutip

| Angka di dokumen ini | Sumber |
|---|---|
| 43.692 / 28.449 / 19 anotator, distribusi label mentah | `reports/01_dataset_identification.md`, log `step3_pipeline_final.ipynb` |
| 43.520 / 171 / 1 unit kosong / 103-150 / 28.448 | log pembersihan `step3_pipeline_final.ipynb`, `data/interim/splits_summary.json` |
| Distribusi label agregat & tie-break (4.414 dst.) | output cell agregasi `step3_pipeline_final.ipynb` |
| Split 19.986/4.229/4.233, 26.157 grup, leakage 0 | `splits_summary.json` + verifikasi ulang `demo/cek_angka_flag.py` |
| 15.166 entri kamus, sitasi W19-3506 | `reports/06_preprocessing_decisions.md`, `README.md` |
| 13 teks flag (5/4/4), anomali `ht`/`kaya` | `src/flag_needs_review.py`, `reports/06_anomali_top_slang_dump.md`, output `demo/cek_angka_flag.py` |
| Checkpoint model, hyperparameter rencana | `demo/bagan_penelitian.py` (figur 22), belum dieksekusi |
