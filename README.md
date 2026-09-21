# Indonesian Hate Speech Analyzer

Proyek Workshop Proyek Sistem Cerdas - Kelompok 5.
Deteksi hate speech berbahasa Indonesia dengan perbandingan IndoBERT (general-domain) vs IndoBERTweet (domain-adaptive media sosial).

## Struktur folder

```
Projek/
├── Dataset/                          # dataset mentah ASLI - jangan pernah diubah
│   ├── indotoxic2024_annotated_data-3.jsonl
│   └── indotoxic2024_annotator_data.jsonl
├── data/
│   ├── external/                     # sumber luar (kamus slang)
│   ├── raw/                          # salinan 1:1 dari Dataset/ (SHA256-verified)
│   ├── interim/                      # hasil antara (metadata agreement, audit split)
│   └── processed/                    # data final siap training (train/val/test.jsonl)
│                                      # + *_preprocessed.jsonl (siap tokenizer)
├── src/                                 # modul python untuk pipeline
│   └── preprocessing.py                 # preprocessing (jalankan: python src/preprocessing.py)
│                                        # modul modeling berikutnya (train.py, evaluate.py) diisi per step
├── notebooks/                           # semua notebook proses, urut per step
│   ├── step1_identifikasi_dataset.ipynb
│   ├── step2_analisis_agregasi.ipynb
│   ├── step3_pipeline_final.ipynb
│   ├── step4_eda.ipynb
│   └── step5_eda_annotator_bias.ipynb
├── models/                              # checkpoint hasil fine-tuning
│   ├── indobert/
│   └── indobertweet/
├── app/                                 # streamlit app (nanti)
├── reports/                             # laporan urut per step + figur
│   ├── figures/                         # figur EDA (16 file, 01 s/d 16)
│   ├── 01_dataset_identification.md
│   ├── 06_preprocessing_decisions.md    # dokumentasi keputusan preprocessing
│   └── step2_prompt.md                  # dokumentasi keputusan step 2-3
├── demo/                                # script demo presentasi (arsip)
├── config.yaml                          # konfigurasi terpusat
├── requirements.txt
└── .gitignore
```

## Cara menjalankan

1. Install dependencies: `pip install -r requirements.txt`
2. Urutan notebook (jalankan berurutan):
   - `notebooks/step1_identifikasi_dataset.ipynb` - verifikasi dataset
   - `notebooks/step2_analisis_agregasi.ipynb` - analisis skema label & agregasi
   - `notebooks/step3_pipeline_final.ipynb` - agregasi + split (membuat data/processed)
   - `notebooks/step4_eda.ipynb` - EDA konten (data final dari data/processed)
   - `notebooks/step5_eda_annotator_bias.ipynb` - EDA bias anotator
3. Preprocessing untuk model:
   - `python src/preprocessing.py` - preprocessing teks dan fitur struktural
     (membuat data/processed/{split}_preprocessed.jsonl; lihat
     `reports/06_preprocessing_decisions.md` untuk dokumentasi keputusan desain)

## Sumber kamus slang

File: `data/external/new_kamusalay.csv` (15.166 entri)
- Sumber: https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv
- Sitasi (dari ACL Anthology, W19-3506):
  Muhammad Okky Ibrohim and Indra Budi. 2019. Multi-label Hate Speech and
  Abusive Language Detection in Indonesian Twitter. In *Proceedings of the
  Third Workshop on Abusive Language Online (ALW3)*, pages 46-57, Florence,
  Italy. Association for Computational Linguistics.
- Link: https://aclanthology.org/W19-3506/
- DOI: https://doi.org/10.18653/v1/W19-3506
- Format: csv dua kolom (`alay`, `normal`). Tidak di-download ulang tiap run
  (disimpan permanen supaya reproducible; kalau mau baru, ganti isi file ini).

## Data

Perhatikan perbedaan penting antara arsip hasil split dan file yang sudah
melewati preprocessing:

- **Sumber kebenaran**: `Dataset/` (43.692 anotasi, 19 anotator) - jangan pernah diubah
- **Arsip hasil split Step 3** (label final, satu baris = satu teks):
  `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`
  Tetap disimpan **sebagai arsip dan referensi** - di sini bisa dilihat
  label final, agreement, tie-break flags, dan group_id dalam bentuk apa adanya.
  Kalau misal desain preprocessing berganti, file ini yang jadi starting point
  untuk regenerate.
- **Dipakai untuk tokenization & training**:
  `data/processed/train_preprocessed.jsonl`, `val_preprocessed.jsonl`, `test_preprocessed.jsonl`
  (kolom `text_clean` yang di-pakai tokenizer BERT, plus kolom `fitur` untuk
  severity scoring atau error analysis nanti, plus kolom biner
  `needs_manual_review` yang menandai teks dengan penggantian slang >= 50 kata
  sebagai target audit manual di bab error analysis).
- File besar dibagikan via Google Drive, sudah di-exclude dari git
