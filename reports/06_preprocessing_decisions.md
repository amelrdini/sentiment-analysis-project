# Laporan Keputusan Desain Preprocessing - Step 6

Dibuat dari pertimbangan temuan EDA Step 4 (masih ada di `notebooks/step4_eda.ipynb`)
dan ditulis bersama module `src/preprocessing.py`.

## Sumber data dan kamus

- Dataset asli    : `Dataset/indotoxic2024_annotated_data-3.jsonl` (43.692 anotasi mentah)
- Dataset olahan  : `data/processed/train.jsonl` / `val.jsonl` / `test.jsonl`
  (hasil split Step 3; ke-3 file sudah zero-leakage antar split, dan tetap
  disimpan sebagai arsip/referensi; file `_preprocessed.jsonl` dihasilkan dari file ini)
- Kamus slang     : `data/external/new_kamusalay.csv`
  - Sumber       : https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv
  - Sitasi       : Muhammad Okky Ibrohim and Indra Budi. 2019. Multi-label Hate
    Speech and Abusive Language Detection in Indonesian Twitter. In
    *Proceedings of the Third Workshop on Abusive Language Online (ALW3)*,
    pages 46-57, Florence, Italy. Association for Computational Linguistics.
  - Link         : https://aclanthology.org/W19-3506/
  - DOI          : https://doi.org/10.18653/v1/W19-3506
  - Jumlah entri bersih : 15.166 (setelah lompat baris kosong + hilangkan duplikat)

## Alur preprocessing yang dipakai

Struktur data yang dihasilkan (per baris, file `{split}_preprocessed.jsonl`):
- `text`           : teks ASLI (masih utuh kapitalnya) untuk analisis manual
- `text_clean`     : teks siap tokenizer (lowercase, slang diganti)
- `fitur`          : 10 fitur struktural hasil analisis EDA Step 4
   (caps_ratio, n_allcaps_word, n_elongated, n_url, n_mention, n_hashtag,
   n_emoji, n_word, n_char, n_slang_replaced)

## Keputusan desain dan trade-off

### 1. Lowercase - simpan sinyal kapital sebagai fitur terpisah

Kapital berkorelasi dengan toxicity (43,1% di toxic vs 36,1% di non-toxic).
Keputusan: **`text_clean` di-lowercase** (konsisten dengan tokenizer IndoBERT /
IndoBERTweet yang memang tidak peka huruf besar-kecil), **tetapi sinyal kapital
disimpan sebelum lowercase sebagai fitur** `caps_ratio` dan `n_allcaps_word`,
supaya bisa dipakai di modul severity scoring atau error analysis tanpa
mengorbankan konsistensi tokenizer.

### 2. Huruf berulang - kompres tapi simpan

Huruf berulang juga berkorelasi dengan toxicity (20,3% vs 13,6%) sebagai
indikasi ekspresi emosi. Keputusan: kompres run huruf 3 ke atas menjadi 2 di
`text_clean` (contoh "anjirrrr" -> "anjirr"), tetapi **simpan jumlahnya
sebagai fitur** `n_elongated` untuk error analysis dan modul severity scoring.

### 3. Hashtag, mention, URL

- **URL**: dihapus seluruhnya dari teks; jumlahnya dicatat sebagai `n_url`.
  Tidak informatif (link mati atau iklan).
- **Mention** (@user): dihapus seluruhnya; jumlahnya dicatat sebagai `n_mention`.
  Mention hanya identitas akun pengguna, bukan konten diskusi.
- **Hashtag**: **simbol # dihapus tetapi katanya dipertahankan** di `text_clean`.
  Alasannya: kata di hashtag sering mengandung informasi topik yang relevan,
  contoh `#AniesMuhaimin2024` atau `#EastTurkistan`. Jumlahnya juga dicatat
  sebagai fitur (`n_hashtag`).

### 4. Emoji - hitung lalu hapus

Emoji tidak dipahami tokenizer BERT dan bisa merusak tokenisasi.
Keputusan: hapus dari `text_clean`, tetapi hitung jumlahnya sebagai `n_emoji`
(21,4% teks memuat emoji, dan EDA menunjukkan emoji muncul lebih sering di toxic).

### 5. Normalisasi slang

Kamus `new_kamusalay.csv` dipakai untuk menyelaraskan kata slang Indonesia
(contoh: "kemaren" -> "kemarin", "bgt" -> "banget", "praktek" -> "praktik").
15.166 entri kamus dipetakan kata-dua-kata (word-by-word), bukan substring match,
agar tidak merusak kata yang kebetulan memuat substring.
Keputusan: **teks asli (`text`) dan teks bersih (`text_clean`) keduanya tetap
disimpan di `{split}_preprocessed.jsonl`** sehingga bisa ditinjau ulang bila
perlu (lihat `data/processed/train_preprocessed.jsonl`).

### 6. Tidak stemming, tidak stopword removal

Tokenizer BERT-style (WordPiece) butuh teks natural berimbuhan. Melakukan
stemming atau stopword removal justru membuang informasi konteks.
Contoh: "menandai" dan "tandai" dilebur oleh stemmer, padahal dalam kalimat
anotasi keduanya punya makna berbeda.

### 7. Kasus anomali mapping kamus yang perlu dicatat

Beberapa mapping kamus bisa merusak konteks. Dua kasus yang kami temukan saat
verifikasi manual teks yang paling banyak terkena penggantian slang:

**a. Mapping `ht` menjadi "hati" (SUDAH DIPERBAIKI)**

Teks `HT: Chinese Taipei` awalnya dipetakan menjadi "hati cina taipei", karena
kamus memuat `ht -> hati`. Dalam konteks media sosial `HT` hampir selalu berarti
`hashtag`, bukan kata `hati` secara harfiah.

Perbaikan: di `src/preprocessing.py`, sebelum kamus dipakai, token `ht` (kata
penuh) di-substitusi menjadi `hashtag` sehingga kamus tidak lagi menimpanya.
Setelah perbaikan ini, jumlah teks yang masih memuat frasa "hati cina" di
train tinggal 1 (dari sebelumnya lebih banyak), dan di val/test sudah 0.
Lihat `src/flag_needs_review.py` untuk angka per split.

**b. Mapping `kaya` menjadi "kayak" (SENGAJA TIDAK DISENTUH)**

Token `kaya` ambigu secara inheren: bisa berarti `kayak` (seperti) atau `kaya`
(kaya/rich). Contoh distorsi ditemukan di teks `4-4735`: `gw g kaya` dimaksudkan
sebagai "saya tidak kaya", tetapi kamus memetakannya menjadi "gue tidak kayak".
Kami **sengaja TIDAK** menambahkan pengecualian untuk kata ini, karena mengubahnya
berisiko merusak ribuan kasus lain di mana `kaya` memang dimaksud sebagai "kayak".
Kata ini dicatat sebagai keterbatasan yang tidak dapat dipulihkan otomatis.

### 8. Kolom metadata `needs_manual_review`

Karena sebagian distorsi konsentrasi di teks dengan penggantian slang sangat
tinggi, kami menambahkan kolom biner `needs_manual_review` pada tiap baris file
`{split}_preprocessed.jsonl`:

- `True`  jika `n_slang_replaced >= 50` (teks kena penggantian slang sangat banyak,
  rawan distorsi makna)
- `False` jika tidak

Jumlah yang ditandai (dari `src/flag_needs_review.py`):
- train: 5 teks
- val  : 4 teks
- test : 4 teks

Kolom ini TIDAK dipakai saat training - hanya sebagai penanda bagi error analysis
atau audit manual. Saat analisis kesalahan nanti, teks ber-`needs_manual_review=True`
bisa diisolasi untuk diperiksa apakah kesalahan model terjadi karena label yang
salah atau karena distorsi preprocessing.

## Verifikasi sampel

10-15 contoh SEBELUM/SESUDAH ditampilkan saat menjalankan
`python src/preprocessing.py`. Setiap baris menunjukkan:
- `text` asli (misal masih kapital, bersama emoji)
- `text_clean` setelah konversi (lowercase, slang diganti, dan sebagainya)
- fitur struktural yang dihasilkan (caps_ratio, n_allcaps_word, n_elongated, dst.)

## Verifikasi manual teks dengan penggantian slang terbanyak

Hasil audit manual atas 5 teks dengan `n_slang_replaced` tertinggi (111, 72, 60,
59, 52) tersimpan di `reports/06_anomali_top_slang_dump.md` (isi lengkap teks
asli vs text_clean berdampingan). Ringkasannya:

| text_id | toxicity | n_ganti | Jenis | Distorsi |
|---|---|---|---|---|
| 4-4735 | 0 | 111 | rant politik | ya (gw g kaya -> gue tidak kayak) |
| 4-2055 | 0 | 72 | katalog e-commerce | ya (sb -> cuma, M -> sama di size chart) |
| 4-1051 | 0 | 60 | rant politik EN/ID | aman |
| 4-1964 | 0 | 59 | daftar username | tidak bermakna (teks asli berantakan) |
| 4-2453 | 0 | 52 | promo kuliner | aman |

Kesimpulan: distorsi nyata terkonsentrasi di teks sangat panjang (rant/katalog)
dan sangat pendek (username). Kelima teks ini semuanya berlabel `toxicity=0`
(konten non-hate), jadi distorsi tidak mengubah prediksi toxic vs non-toxic.
Hanya dua mapping rawan yang diidentifikasi: `ht` (sudah diperbaiki) dan `kaya`
(dibiarkan, dicatat sebagai keterbatasan).

## Batasan

1. `text_clean` sudah lowercase, tapi `text` tetap asli - kalau suatu saat
   tokenizer membutuhkan teks asli dengan kapital (contoh IndoBERTweet yang
   tokenizer-nya memang peka terhadap struktur Twitter), tinggal pakai
   kolom `text`.
2. Kamus slang hanya mencocokkan kata yang persis ada di kamus; slang baru
   tidak akan dikenali. Bisa ditambahkan di iterasi berikutnya.
3. 15.166 entri kamus ini bukan kamus tuntas; masih ada kata-kata slang yang
   belum ternormalisasi.
4. Kata ambigu seperti `kaya` dan `ht` bisa memetakan konteks yang salah.
   Untuk `ht` sudah ada pengecualian khusus; untuk `kaya` sengaja dibiarkan
   karena mengubahnya berisiko merusak kasus lain. Teks yang paling terdampak
   ditandai lewat kolom `needs_manual_review`.

Kalau kelompok menemukan kamus yang lebih lengkap, tinggal ganti isi
`data/external/new_kamusalay.csv` dan jalankan ulang module ini.
