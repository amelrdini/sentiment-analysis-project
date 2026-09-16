# Resume Dataset — IndoToxic2024

## Informasi Dataset

- **Nama dataset:** IndoToxic2024 (berdasarkan nama file)
- **Sumber publikasi:** Genta Indra Winata et al., 2024
- **Domain:** Hate speech & toxicity berbahasa Indonesia dari media sosial
- **Fokus multi-aspek:** toxicity, profanity, threats, insults, identity attack, sexually explicit

---

## Daftar File

| File | Ukuran | Format | Jumlah Baris |
|---|---|---|---|
| `indotoxic2024_annotated_data-3.jsonl` | ~34,6 MB | JSONL | 43.692 |
| `indotoxic2024_annotator_data.jsonl` | ~4,8 KB | JSONL | 19 |

> Tidak ada file README atau dokumentasi bawaan lainnya di folder ini.

---

## Skema File 1: `indotoxic2024_annotated_data-3.jsonl`

Setiap baris adalah satu objek JSON dengan 14 key berikut:

| Key | Tipe | Deskripsi |
|---|---|---|
| `batch_id` | string | ID batch pengumpulan data (1, 2, 3, 4, 101, 102, 103) |
| `batch_text_id` | string | Nomor urut teks dalam batch |
| `text_id` | string | ID unik teks (gabungan batch_id + batch_text_id, misal `"2-1"`) |
| `metadata_id` | string | ID sumber/metadata asal teks |
| `annotator_id` | string | ID anotator yang menilai teks ini (1–21, tanpa 14 & 17) |
| `text` | string | Konten teks dari media sosial |
| `initial_paragraph` | string | Paragraf awal/konteks jika teks adalah reply (kosong jika tidak ada) |
| `topic` | string | Topik teks, multi-label dipisah koma (contoh: `"Terpolarisasi, Jewish"`) |
| `is_noise_or_spam_text` | int (0/1) | Flag: apakah teks adalah noise/spam |
| `related_to_election_2024` | int (0/1) | Flag: apakah teks terkait Pemilu 2024 |
| `toxicity` | int (0/1) | **Label utama:** apakah teks mengandung toxic/hate speech |
| `profanity_obscenity` | int (0/1) | Mengandung kata kasar/pornografi |
| `threat_incitement_to_violence` | int (0/1) | Mengandung ancaman/hasutan kekerasan |
| `insults` | int (0/1) | Mengandung hinaan |
| `identity_attack` | int (0/1) | Serangan berdasarkan identitas (SARA) |
| `sexually_explicit` | int (0/1) | Konten seksual eksplisit |

### Contoh Data

```json
{"batch_id": "2", "batch_text_id": "4", "text_id": "2-4", "metadata_id": "98620", "annotator_id": "20", "text": "Gua ragu Israel (\"negara\") yang kita lihat sekarang di TV, sosmed, dll itu adalah Israel yang dimaksud di Alkitab Kenapa bela genosida mati-matian? Bukannya tuhan mengajar kasihilah sesamamu seperti kau mengasihi dirimu sendiri?", "initial_paragraph": "", "topic": "Terpolarisasi, Jewish", "is_noise_or_spam_text": 0, "related_to_election_2024": 0, "toxicity": 1, "profanity_obscenity": 0, "threat_incitement_to_violence": 0, "insults": 0, "identity_attack": 1, "sexually_explicit": 0}
```

---

## Skema File 2: `indotoxic2024_annotator_data.jsonl`

Setiap baris adalah profil demografis satu anotator (total 19 anotator).

| Key | Tipe | Deskripsi |
|---|---|---|
| `annotator_id` | int | ID anotator (1–21, tanpa 14 & 17) |
| `ethnicity` | string | Suku/etnis (Jawa, Tionghoa, Batak, Minang, Madura, Arab, Bali) |
| `religion` | string | Agama (Islam, Kristen, Buddha, Hindu, Syiah, Ahmadiyah, Kepercayaan Lokal) |
| `disability` | string | Apakah penyandang disabilitas (`"yes"` / `"no"`) |
| `lgbt` | string | Apakah LGBTQ+ (`"yes"` / `"no"`) |
| `gender` | string | Jenis kelamin (`"F"` / `"M"`) |
| `age` | int | Usia (20–50) |
| `city` | string | Kota domisili |
| `last_education_degree` | string | Pendidikan terakhir |
| `job_status` | string | Status pekerjaan |
| `president vote leaning` | string | Kecenderungan pilihan presiden (1, 2, 3, unknown, Abstain) |

### Contoh Data

```json
{"annotator_id": 1, "ethnicity": "Tionghoa", "religion": "Kristen", "disability": "no", "lgbt": "no", "gender": "F", "age": 50, "city": "Jakarta", "last_education_degree": "Diploma", "job_status": "Bekerja", "president vote leaning": "2"}
```

---

## Struktur Anotasi

- **1 baris = 1 anotasi** (satu teks dinilai oleh satu anotator), **bukan** 1 baris = 1 teks.
- Dari 43.692 baris, terdapat **28.449 teks unik** (`text_id`). Sebagian teks dinilai oleh lebih dari satu anotator:

| Jumlah anotasi per teks | Jumlah teks |
|---|---|
| 1 | 18.039 |
| 2 | 9.864 |
| 3 | 96 |
| 5 | 1 |
| 10 | 5 |
| 11 | 95 |
| 13 | 349 |

- Belum ada agregasi label (belum ada label mayoritas voting). Perlu diproses untuk mendapatkan 1 label final per teks.

---

## Distribusi Label

| Label | 0 | 1 |
|---|---|---|
| `toxicity` | 36.793 | **6.899** (15,8%) |
| `profanity_obscenity` | 42.398 | 1.294 (3,0%) |
| `threat_incitement_to_violence` | 42.188 | 1.504 (3,4%) |
| `insults` | 40.470 | 3.222 (7,4%) |
| `identity_attack` | 40.495 | 3.197 (7,3%) |
| `sexually_explicit` | 43.457 | 235 (0,5%) |
| `is_noise_or_spam_text` | 40.755 | 2.937 (6,7%) |
| `related_to_election_2024` | 38.870 | 4.822 (11,0%) |

---

## Topik

Nilai unik `topic` (multi-label, comma-separated):

- Disabilitas, Jewish, Kristen, LGBTQ+, Rohingya, Syiah, Ahmadiyah, Terpolarisasi, Tionghoa
- Ditambah nilai anomali: `"1"` dan `"UNKNOWN"`

---

## Catatan

- **2 baris memiliki `text` kosong** (terverifikasi): baris fisik **1256** (`text_id: 2-1256`, `annotator_id: 21`) dan baris fisik **28394** (`text_id: 103-150`, `annotator_id: 6`). Nilai `text` sebenarnya adalah spasi (`" "`, `"  "`), bukan null. Menariknya, keduanya juga berlabel `topic: "UNKNOWN"`.
- `initial_paragraph` terisi pada 4.210 baris (teks berupa reply/komentar dengan konteks).
- 19 anotator: 12 perempuan, 7 laki-laki.
- Dataset ini cocok untuk tugas **klasifikasi hate speech multi-label** maupun **binary classification** (toxicity) dengan pendekatan Transformer seperti IndoBERT dan IndoBERTweet.