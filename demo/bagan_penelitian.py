"""Bagan rancangan metode penelitian Kelompok 5 - deteksi hate speech Indonesia.

Menghasilkan:
  reports/figures/22_bagan_penelitian.png  (bagan alur utama, 9 tahap)
  reports/figures/23_matriks_evaluasi.png  (formula metrik + protokol evaluasi)

Semua angka diambil dari artefak terverifikasi:
  data/interim/splits_summary.json, reports/01_dataset_identification.md,
  reports/06_preprocessing_decisions.md
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.family"] = "DejaVu Sans"

MERAH = "#C0392B"
MERAH_MUDA = "#FDEDEC"
ORANGE = "#E67E22"
ORANGE_MUDA = "#FDF2E9"
BIRU = "#2471A3"
BIRU_MUDA = "#EBF5FB"
HIJAU = "#1E8449"
HIJAU_MUDA = "#EAFAF1"
UNGU = "#6C3483"
UNGU_MUDA = "#F4ECF7"
ABU_TUA = "#2C3E50"
ABU = "#5D6D7E"
KUNING_MUDA = "#FEF9E7"
TEPI_KUNING = "#B7950B"

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def kotak(ax, x, y, w, h, judul, baris, warna_tepi, warna_isi,
          fs_judul=10.5, fs_isi=8.3, lw=1.8, alpha_judul=0.14):
    """Kotak bulat dengan judul berlatar tipis + daftar baris."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=warna_tepi, facecolor=warna_isi)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.16, judul, ha="center", va="top",
            fontsize=fs_judul, fontweight="bold", color=warna_tepi)
    ax.add_patch(FancyBboxPatch(
        (x, y + h - 0.34), w, 0.32,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0, facecolor=warna_tepi, alpha=alpha_judul))
    ax.text(x + w / 2, y + h - 0.18, judul, ha="center", va="center",
            fontsize=fs_judul, fontweight="bold", color="white",
            bbox=dict(boxstyle="square,pad=0", fc=warna_tepi, ec="none"))
    ax.text(x + w / 2, y + h - 0.44, baris, ha="center", va="top",
            fontsize=fs_isi, color=ABU_TUA, linespacing=1.45)


def panah(ax, x1, y1, x2, y2, warna=ABU, style="-|>", lw=2.0, ms=16):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, color=warna))


def label_samping(ax, x, y, teks, warna):
    ax.text(x, y, teks, ha="center", va="center", fontsize=7.6, color=warna,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=warna, lw=1.0))


# ------------------------------------------------------------------ bagan utama
def bagan_penelitian():
    fig, ax = plt.subplots(figsize=(15.5, 19.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 129)
    ax.axis("off")

    ax.text(50, 127.6, "BAGAN RANCANGAN METODE PENELITIAN",
            ha="center", fontsize=17, fontweight="bold", color=ABU_TUA)
    ax.text(50, 125.4,
            "Perbandingan Fine-Tuning IndoBERT vs IndoBERTweet untuk Klasifikasi "
            "Multi-Label Hate Speech Berbahasa Indonesia (IndoToxic2024)",
            ha="center", fontsize=10.5, color=ABU, style="italic")
    ax.text(50, 123.6,
            "Workshop Proyek Sistem Cerdas - Kelompok 5  |  seed 42  |  split 70 : 15 : 15  |  "
            "agregasi majority + safety-first (v/n >= 0,5)",
            ha="center", fontsize=8.8, color=ABU)

    # ================= TAHAP 1 - DATASET =================
    kotak(ax, 27, 114.2, 46, 7.4, "1  DATASET : INDOTOXIC2024 (DATA ASLI)",
          "indotoxic2024_annotated_data-3.jsonl  (34,6 MB, JSONL)\n"
          "43.692 anotasi  |  19 anotator  |  28.449 teks unik (text_id)\n"
          "6 label biner per anotasi: toxicity, profanity_obscenity,\n"
          "threat_incitement_to_violence, insults, identity_attack,\n"
          "sexually_explicit   +  metadata topik & demografi anotator",
          MERAH, MERAH_MUDA, fs_judul=11)
    label_samping(ax, 79.5, 117.9,
                  "Sumber: Genta Indra Winata\net al., 2024 (media sosial\nberbahasa Indonesia)",
                  MERAH)

    panah(ax, 50, 114.1, 50, 112.4)

    # ================= TAHAP 2 - VALIDASI & PEMBERSIHAN =================
    kotak(ax, 27, 105.2, 46, 7.0, "2  VALIDASI & PEMBERSIHAN DATA",
          "Salin 1:1 ke data/raw (SHA256-verified)\n"
          "43.692 -> 43.520 baris valid (buang 171 baris konten minoritas\n"
          "< 10 karakter & 1 baris teks kosong: text_id 2-1256)\n"
          "1 baris teks kosong direkonstruksi (103-150)  |  notebook step 1-2",
          ORANGE, ORANGE_MUDA)
    panah(ax, 50, 105.1, 50, 103.4)

    # ================= TAHAP 3 - AGREGASI =================
    kotak(ax, 27, 96.4, 46, 6.8, "3  AGREGASI LABEL (MULTI-ANOTATOR)",
          "majority_safety_first: label positif jika v/n >= 0,5\n"
          "(tie di-resolve ke positif untuk semua 6 label)\n"
          "43.520 anotasi -> 28.448 unit teks berlabel final\n"
          "26.157 grup idiomatik unik (group_id)  |  notebook step 3",
          ORANGE, ORANGE_MUDA)
    panah(ax, 50, 96.3, 50, 94.6)

    # ================= TAHAP 4 - SPLIT =================
    kotak(ax, 27, 87.2, 46, 7.2, "4  GROUP-AWARE MULTI-LABEL SPLIT",
          "Iterative Stratification (scikit-multilearn) per group_id\n"
          "grup tidak boleh lintas split -> mencegah leakage (verifikasi: 0)\n"
          "TRAIN 19.986 teks (15.150 grup)  |  VAL 4.229 (3.186 grup)\n"
          "TEST 4.233 (3.158 grup)  |  rasio 70 : 15 : 15  |  seed 42",
          HIJAU, HIJAU_MUDA)
    panah(ax, 50, 87.1, 50, 85.4)

    # ================= TAHAP 5 - EDA =================
    kotak(ax, 27, 78.4, 46, 6.8, "5  EDA (SEBELUM & SESUDAH PREPROCESSING)",
          "Distribusi label & co-occurrence 6 label; profil 19 anotator\n"
          "& demographic bias; panjang teks; kosakata & wordcloud\n"
          "toxic vs non-toxic (teks asli + text_clean);\n"
          "notebook step 4, 5, 6  ->  21 figur di reports/figures/",
          UNGU, UNGU_MUDA)
    panah(ax, 50, 78.3, 50, 76.6)

    # ================= TAHAP 6 - PREPROCESSING =================
    kotak(ax, 27, 68.6, 46, 7.8, "6  PREPROCESSING TEKS  (src/preprocessing.py)",
          "Lowercase  |  kompres huruf berulang (anjirrrr -> anjirr)\n"
          "URL & mention @user dihapus (n_url, n_mention dicatat)\n"
          "hashtag: simbol # dibuang, kata dipertahankan; emoji dihitung lalu dihapus\n"
          "Normalisasi slang kamus new_kamusalay.csv (15.166 entri, word-by-word)\n"
          "TANPA stemming & TANPA stopword removal (konteks penting utk BERT)\n"
          "Fitur struktural: n_kata, n_char, rasio_huruf_kapital, dll.  ->  text_clean",
          BIRU, BIRU_MUDA)
    label_samping(ax, 79.5, 72.5,
                  "Output: {split}_preprocessed.jsonl\n13 teks flag needs_manual_review\n(penggantian slang >= 50 kata)",
                  BIRU)
    panah(ax, 50, 68.5, 50, 66.8)

    # ================= TAHAP 7 - MODELING =================
    kotak(ax, 3.5, 47.5, 44, 18.6, "7a  INDOBERT (BASELINE GENERAL-DOMAIN)",
          "indobenchmark/indobert-base-p1\n"
          "Pre-training: 220 juta kata Wikipedia+ artikel\n"
          "berbahasa Indonesia (IndoLEM / IndoNLU)\n"
          "12 layer, 768 hidden, 12 head, ~124 juta parameter\n"
          "vocab: WordPiece 50.000 (casing: p1 = uncased)\n"
          "\n"
          "Fine-tuning:\n"
          "- input: text_clean -> tokenizer IndoBERT\n"
          "- max_length = 128 token, padding/truncation\n"
          "- 6 unit output sigmoid (BCE loss, multi-label)\n"
          "- optimizer AdamW, lr 2e-5, weight decay 0,01\n"
          "- batch 32, epoch 4 (early stop dgn val macro-F1)\n"
          "- seed 42, checkpoint terbaik disimpan di models/indobert/",
          BIRU, BIRU_MUDA, fs_isi=8.0)

    kotak(ax, 52.5, 47.5, 44, 18.6, "7b  INDOBERTWEET (DOMAIN-ADAPTIVE)",
          "indolem/indobertweet-base-uncased\n"
          "Pre-training lanjutan IndoBERT pada 286 juta tweet\n"
          "berbahasa Indonesia (Koto et al., IndoLEM)\n"
          "12 layer, 768 hidden, 12 head, ~124 juta parameter\n"
          "vocab: WordPiece 31.923 (uncased, khas media sosial)\n"
          "\n"
          "Fine-tuning: protokol identik dengan 7a:\n"
          "- max_length = 128 token, 6 sigmoid + BCE\n"
          "- AdamW lr 2e-5, batch 32, epoch 4 (early stop)\n"
          "- seed 42, checkpoint di models/indobertweet/\n"
          "Hipotesis: lebih baik menangkap slang, typo,\n"
          "dan elision khas Twitter/X media sosial",
          HIJAU, HIJAU_MUDA, fs_isi=8.0)

    panah(ax, 50, 66.7, 27, 66.7)
    panah(ax, 50, 66.7, 73, 66.7)
    panah(ax, 25.5, 47.4, 25.5, 44.6)
    panah(ax, 74.5, 47.4, 74.5, 44.6)

    # ================= TAHAP 8 - EVALUASI =================
    kotak(ax, 27, 33.0, 46, 11.4, "8  EVALUASI (TEST SET - DIHOLD-OUT)",
          "metrik per label + macro/average:\n"
          "Precision, Recall, F1-score, Support  |  macro & micro average\n"
          "Subset accuracy (exact match 6 label)  |  Hamming loss\n"
          "ROC-AUC & PR-AUC per label (threshold 0,5)\n"
          "Confusion matrix per label + error analysis kualitatif\n"
          "Pengujian signifikansi antar model (bootstrap / McNemar)\n"
          "library: scikit-learn 1.4.2, torch, transformers  |  notebook step 7",
          ABU, "#F2F3F4")
    panah(ax, 50, 32.9, 50, 31.2)

    # ================= TAHAP 9 - ANALISIS & DEPLOY =================
    kotak(ax, 27, 21.0, 46, 10.0, "9  ANALISIS ERROR & DEPLOYMENT",
          "Error analysis per label + teks flag needs_manual_review (13 teks)\n"
          "Studi kasus: slang tak dikenal kamus, ironi/sarkasme\n"
          "Model terbaik -> app/ Streamlit (inference teks bebas)\n"
          "Laporan akhir + notebook reproducible (seed 42, requirements.txt)",
          TEPI_KUNING, KUNING_MUDA)

    # ================= garis sisi: sumber terbuka =================
    kotak(ax, 3.5, 96.4, 18, 25.0, "SUMBER & ALAT",
          "python 3.11\n"
          "pandas 2.1.4\n"
          "numpy 1.26.4\n"
          "matplotlib 3.7.5\n"
          "seaborn 0.13.2\n"
          "scikit-learn 1.4.2\n"
          "scikit-multilearn\n"
          "wordcloud 1.9.6\n"
          "torch\n"
          "transformers (HF)\n"
          "jupyter\n"
          "streamlit\n\n"
          "semua notebook & skrip\n"
          "reproducible, seed 42",
          ABU, "#FBFCFC", fs_judul=9.5, fs_isi=7.8)

    kotak(ax, 78.5, 96.4, 18, 25.0, "OUTPUT DATA",
          "data/raw/\n"
          "  (arsip SHA256)\n"
          "data/interim/\n"
          "  metadata agreement\n"
          "  audit split\n"
          "data/processed/\n"
          "  train/val/test.jsonl\n"
          "  *_preprocessed.jsonl\n"
          "reports/figures/\n"
          "  21 figur EDA\n"
          "models/\n"
          "  indobert/ +\n"
          "  indobertweet/",
          ABU, "#FBFCFC", fs_judul=9.5, fs_isi=7.8)

    # connector tipis dari tahap ke sisi
    for y in (110.0, 101.0, 93.0, 83.0):
        panah(ax, 27, y, 21.6, y, warna="#BDC3C7", lw=1.1, ms=10, style="-|>")
    for y in (72.5, 55.0):
        panah(ax, 73.2, y, 78.4, y, warna="#BDC3C7", lw=1.1, ms=10, style="-|>")

    fig.savefig(os.path.join(FIG_DIR, "22_bagan_penelitian.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("OK  22_bagan_penelitian.png")


# ------------------------------------------------------------------ matriks evaluasi
def matriks_evaluasi():
    fig, ax = plt.subplots(figsize=(15.5, 11.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 76)
    ax.axis("off")

    ax.text(50, 74.5, "MATRIKS EVALUASI MODEL KLASIFIKASI MULTI-LABEL",
            ha="center", fontsize=15.5, fontweight="bold", color=ABU_TUA)
    ax.text(50, 72.4,
            "Rumus, mode rata-rata, dan protokol evaluasi - "
            "klasifikasi multi-label 6 label, IndoToxic2024",
            ha="center", fontsize=10, color=ABU, style="italic")

    # ---------- kolom definisi dasar ----------
    kotak(ax, 2.5, 52.5, 30, 16.5, "KONFUSI MATRIX (PER LABEL)",
          "per label biner (one-vs-rest):\n"
          "TP  - prediksi positif, benar\n"
          "FP  - prediksi positif, salah\n"
          "FN  - prediksi negatif, salah\n"
          "TN  - prediksi negatif, benar\n"
          "\n"
          "di-evaluasi terpisah untuk 6 label:\n"
          "toxicity, profanity_obscenity,\n"
          "threat_incitement_to_violence,\n"
          "insults, identity_attack,\n"
          "sexually_explicit",
          MERAH, MERAH_MUDA, fs_isi=8.2)

    # ---------- kolom rumus utama ----------
    kotak(ax, 35.5, 52.5, 30, 16.5, "RUMUS METRIK UTAMA",
          "Precision = TP / (TP + FP)\n"
          "Recall    = TP / (TP + FN)\n"
          "F1-score  = 2 (P x R) / (P + R)\n"
          "\n"
          "Accuracy   = (TP + TN) / (TP + TN\n"
          "                        + FP + FN)\n"
          "Hamming Loss = rata-rata proporsi\n"
          "  label salah per sampel (6 label)\n"
          "Subset Accuracy = proporsi sampel\n"
          "  dengan SEMUA 6 label benar",
          BIRU, BIRU_MUDA, fs_isi=8.2)

    # ---------- kolom AUC ----------
    kotak(ax, 68.5, 52.5, 29, 16.5, "METRIK BERBASIS THRESHOLD-FREE",
          "ROC-AUC per label:\n"
          "  luas di bawah kurva TPR vs FPR\n"
          "  saat threshold sigmoid digeser\n"
          "PR-AUC per label (average_precision):\n"
          "  luas di bawah kurva Precision vs\n"
          "  Recall - lebih informatif utk label\n"
          "  langka (sexually_explicit 0,6%)\n"
          "Threshold operasional default 0,5\n"
          "(dapat dikalibrasi ulang per label)",
          HIJAU, HIJAU_MUDA, fs_isi=8.2)

    # ---------- mode averaging ----------
    kotak(ax, 2.5, 33.0, 47, 17.0, "MODE RATA-RATA (AVERAGING) MULTI-LABEL",
          "MACRO   : hitung metrik per label lalu rata-rata tanpa bobot\n"
          "  -> semua label setara; sensitif thd label langka\n"
          "  -> METRIK UTAMA pemilihan model (macro-F1)\n"
          "MICRO   : agregat semua TP/FP/FN lintas label lalu hitung\n"
          "  -> didominasi label mayoritas (toxicity 15-16%)\n"
          "WEIGHTED: rata-rata F1 per label, dibobot support kelas\n"
          "SAMPLES : rata-rata metrik per teks (per sampel)\n"
          "semua mode dihitung scikit-learn metrics:\n"
          "  precision_recall_fscore_support(average=...)",
          ORANGE, ORANGE_MUDA, fs_isi=8.2)

    # ---------- protokol ----------
    kotak(ax, 52.5, 33.0, 45, 17.0, "PROTOKOL EVALUASI",
          "1. Evaluasi HANYA di test set (4.233 teks) yang tidak pernah\n"
          "    disentuh training & tuning (hold-out, zero-leakage grup)\n"
          "2. classification_report sklearn -> tabel per label + rata-rata\n"
          "3. multilabel_confusion_matrix -> 6 confusion matrix (2x2)\n"
          "4. roc_auc_score & average_precision_score per label\n"
          "5. Bandingkan IndoBERT vs IndoBERTweet pada test set yang SAMA\n"
          "6. Uji signifikansi: bootstrap 1.000 resample (CI 95% macro-F1)\n"
          "    + McNemar test per label pada prediksi benar/salah\n"
          "7. Error analysis kualitatif: FP/FN terbanyak + 13 teks flag",
          ABU, "#F2F3F4", fs_isi=8.2)

    # ---------- library & environment ----------
    kotak(ax, 2.5, 12.0, 95, 18.5, "LIBRARY, MODEL, & KONFIGURASI EKSPERIMEN",
          "library              : transformers (HuggingFace) AutoTokenizer + AutoModelForSequenceClassification,\n"
          "                       torch (PyTorch, training & inference), scikit-learn 1.4.2 (semua metrik),\n"
          "                       pandas / numpy (agregasi hasil), matplotlib (visualisasi)\n"
          "mode task            : sequence classification MULTI-LABEL - 6 output sigmoid independen (bukan softmax),\n"
          "                       loss Binary Cross-Entropy per label (BCEWithLogitsLoss)\n"
          "model A              : indobenchmark/indobert-base-p1   (vocab 50.000, ~124 juta parameter)\n"
          "model B              : indolem/indobertweet-base-uncased (vocab 31.923, ~124 juta parameter)\n"
          "input                : text_clean, max_length 128 token, padding + truncation\n"
          "training             : AdamW lr 2e-5, batch 32, maks 4 epoch, early stopping (val macro-F1), seed 42\n"
          "evaluasi             : threshold 0,5; classification_report + macro/micro/weighted F1 + Hamming + subset accuracy",
          TEPI_KUNING, KUNING_MUDA, fs_isi=8.2)

    # panah antar panel
    panah(ax, 32.7, 60.5, 35.3, 60.5, warna="#BDC3C7", lw=1.4, ms=11)
    panah(ax, 65.7, 60.5, 68.3, 60.5, warna="#BDC3C7", lw=1.4, ms=11)
    panah(ax, 50, 52.3, 50, 50.2, warna="#BDC3C7", lw=1.4, ms=11)
    panah(ax, 50, 32.8, 50, 30.7, warna="#BDC3C7", lw=1.4, ms=11)

    fig.savefig(os.path.join(FIG_DIR, "23_matriks_evaluasi.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("OK  23_matriks_evaluasi.png")


if __name__ == "__main__":
    bagan_penelitian()
    matriks_evaluasi()
