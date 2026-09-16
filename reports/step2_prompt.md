================================================================================
PENJELASAN METRIK & PENDEKATAN TEKNIS SEBELUM EKSEKUSI FINAL
================================================================================

1. KLARIFIKASI PERBEDAAN JUMLAH TOXICITY POSITIF (2.404 vs 4.236)
--------------------------------------------------------------------------------
Perbedaan angka ini disebabkan oleh perbedaan kriteria ambang batas (threshold) 
agregasi label antara sesi sebelumnya dan sesi ini:

  a. Sesi Sebelumnya (4.236 sampel positif):
     - Menggunakan pendekatan ANY-POSITIVE (OR logic / Union).
     - Sebuah sampel dihitung Positif Toxicity jika minimal ADA 1 ANNOTATOR 
       yang memberikan label 1 (Annotator_1 == 1 OR Annotator_2 == 1 OR Annotator_3 == 1).
     - Menangkap seluruh data berlabel ambigu/borderline sehingga angkanya tinggi.

  b. Sesi Ini (2.404 sampel positif):
     - Menggunakan kriteria MAJORITY VOTING murni (>= 2 dari 3 annotator setuju).
     - Teks baru dianggap Positif Toxicity jika minimal 2 annotator memberi nilai 1.
     - Mengabaikan 1.832 sampel "single-vote positive" (hanya 1 annotator yang menilai 1).

  Verifikasi Rekonsiliasi Data:
  ------------------------------------------------------------------
  - Majority Positive (>= 2 vote = 1)   : 2.404 sampel
  - Single Positive   (tepat 1 vote = 1): 1.832 sampel
  - Total Any Positive (>= 1 vote = 1)  : 4.236 sampel (Data Sesi Lalu)
  ------------------------------------------------------------------
  Kesimpulan: Data dasar 100% konsisten. Perbedaan murni berasal dari pergeseran 
  definisi dari ANY-POSITIVE (>=1) ke MAJORITY VOTING (>=2).


2. KONFIRMASI METODE AGREGASI & TIE-BREAKING
--------------------------------------------------------------------------------
- Konsensus: Menggunakan Majority Voting + Safety-First.
- Skop Tie-Breaking: Berlaku penuh untuk SELURUH 6 LABEL (toxic, severe_toxic, 
  obscene, threat, insult, identity_hate / sexually_explicit).
- Tracking Flag: Kolom `tie_break_applied_<label>` disimpan sebagai boolean flag 
  per label per teks untuk kebutuhan Ablation Study mendatang.


3. KONFIRMASI PEMBERSIHAN & REKONSTRUKSI DATA
--------------------------------------------------------------------------------
- Modus Konten: Menerapkan teks modus untuk `text_id` yang memiliki variasi string.
- Pemulihan Teks Kosong: Imputasi teks kosong/NaN menggunakan referensi `text_id` yang valid.
- Normalisasi Topik: Menyeragamkan penamaan topik anomali di seluruh entri.


4. PENDEKATAN TEKNIS: GROUP-AWARE MULTI-LABEL ITERATIVE STRATIFICATION (STEP 3)
--------------------------------------------------------------------------------
Tantangan: Mencegah data leakage akibat 15.5% data duplikat/near-duplicate sambil 
tetap menjaga keseimbangan distribusi 6 label multi-label di Train, Val, dan Test.

Solusi Algoritma (Group-Level Iterative Stratification):

  1. Grouping ID Assignment:
     Setiap konten identik/near-duplicate dikelompokkan ke dalam satu `group_id` 
     unik. Semua teks yang sama dipastikan memiliki `group_id` yang identik.

  2. Agregasi Label di Tingkat Group:
     Untuk setiap `group_id`, buat satu vektor label multi-label gabungan 
     menggunakan operasi Logical OR (atau Max) dari seluruh anggota di dalam grup. 
     Artinya, 1 grup diwakili oleh 1 baris agregat multi-label.

  3. Iterative Stratification di Tingkat Group:
     Jalankan algoritma Multilabel Stratified Split (misalnya `MultilabelStratifiedKFold` 
     atau `iterative_train_test_split` dari `scikit-multilearn`) pada tingkat **`group_id`**.
     Ini menjamin pemisahan train/val/test secara seimbang untuk 6 label secara berkelanjutan.

  4. Re-mapping ke Individual Samples:
     Petakan kembali hasil pembagian `group_id` ke seluruh baris data asli. 
     Dengan cara ini, seluruh anggota dari satu `group_id` dijamin 100% masuk ke 
     split yang sama (Train, Val, atau Test), menghapus risiko data leakage.

================================================================================
SETELAH DICONFIRM, SIAP DIEKSEKUSI UNTUK PENYIMPANAN KE `data/processed/`.
================================================================================