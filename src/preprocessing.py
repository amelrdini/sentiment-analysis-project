"""Preprocessing teks IndoToxic2024 untuk tahap modeling.

Jalankan dengan: python src/preprocessing.py
Sumber data    : data/processed/{split}.jsonl
Hasil          : data/processed/{split}_preprocessed.jsonl

Keputusan desain (rincian di reports/06_preprocessing_decisions.md):
- dua kolom teks: text (asli, kapital utuh) dan text_clean (siap tokenizer)
- sinyal kapital & huruf berulang disimpan sebagai fitur terpisah sebelum dibersihkan
- hashtag: hapus simbol # tapi pertahankan katanya
- mention dan URL dihapus seluruhnya
- emoji dihapus dari teks tetapi jumlahnya dicatat sebagai fitur
- huruf berulang 3+ dikompres ke 2, jumlah kata berulang dicatat sebagai fitur
- normalisasi slang memakai data/external/new_kamusalay.csv (Ibrohim & Budi 2019)
- tidak ada stemming maupun stopword removal agar teks tetap natural untuk tokenizer
"""

import os
import re
import csv
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# === dua path ini yang perlu diubah kalau folder pindah ===
DATA = r'C:\SEMESTER 5\Project Sistem Cerdas\Projek\data\processed'
KAMUS = r'C:\SEMESTER 5\Project Sistem Cerdas\Projek\data\external\new_kamusalay.csv'
# ==========================================================

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#(\w+)')
REPEAT_RE = re.compile(r'(.)\1{2,}')
EMOJI_RE = re.compile(
    '[\U0001F000-\U0001FAFF\u2600-\u27BF\u2190-\u21FF\u2B00-\u2BFF\uFE00-\uFE0F]')
SPASI_RE = re.compile(r'\s+')

LABEL_COLS = ['toxicity', 'profanity_obscenity', 'threat_incitement_to_violence',
              'insults', 'identity_attack', 'sexually_explicit']


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def load_kamus(path):
    """Baca csv format alay,normal; lompat baris kosong; key di-lowercase."""
    kamus = {}
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            kunci = row[0].strip().lower()
            nilai = row[1].strip()
            if kunci and nilai and kunci not in kamus:
                kamus[kunci] = nilai
    return kamus


def hitung_fitur(teks):
    """Semua fitur dihitung dari teks asli sebelum ada perubahan apa pun."""
    daftar_kata = teks.split()
    n_huruf = sum(1 for ch in teks if ch.isalpha())
    n_upper = sum(1 for ch in teks if ch.isupper())
    return {
        'caps_ratio': round(n_upper / n_huruf, 4) if n_huruf > 0 else 0.0,
        'n_allcaps_word': sum(1 for w in daftar_kata
                              if len(w) >= 3 and w.isupper()),
        'n_elongated': sum(1 for w in daftar_kata
                           if re.search(r'(.)\1{2,}', w)),
        'n_url': len(URL_RE.findall(teks)),
        'n_mention': len(MENTION_RE.findall(teks)),
        'n_hashtag': len(HASHTAG_RE.findall(teks)),
        'n_emoji': len(EMOJI_RE.findall(teks)),
        'n_word': len(daftar_kata),
        'n_char': len(teks),
    }


def bersihkan(teks, kamus):
    """Kembalikan teks siap tokenizer beserta jumlah kata slang yang diganti."""
    teks = URL_RE.sub(' ', teks)
    teks = MENTION_RE.sub(' ', teks)
    teks = HASHTAG_RE.sub(r'\1', teks)
    teks = EMOJI_RE.sub(' ', teks)
    teks = REPEAT_RE.sub('\\1\\1', teks)
    teks = teks.lower()
    # pengecualian khusus: ht di medsos berarti hashtag, bukan hati
    # (tanpa ini, kamus memetakan ht -> hati dan merusak konteks)
    teks = re.sub(r'\bht\b', 'hashtag', teks)

    n_ganti = 0
    hasil_kata = []
    for w in teks.split():
        kunci = w.strip('.,"!?():;\'')
        if kunci in kamus:
            hasil_kata.append(kamus[kunci])
            n_ganti += 1
        else:
            hasil_kata.append(w)
    teks = ' '.join(hasil_kata)
    teks = SPASI_RE.sub(' ', teks).strip()
    return teks, n_ganti


def pilih_sampel(rows, jumlah=15):
    """Ambil sampel beragam: kapital tinggi, berulang, hashtag, slang, dan biasa."""
    kategori = {
        'KAPITAL': [], 'BERULANG': [], 'HASHTAG': [], 'SLANG': [], 'BIASA': [], 'URL/MENTION': []
    }
    for r in rows:
        f_ = r['fitur']
        if f_['caps_ratio'] >= 0.25 and len(kategori['KAPITAL']) < 4:
            kategori['KAPITAL'].append(r)
        elif f_['n_elongated'] >= 2 and len(kategori['BERULANG']) < 3:
            kategori['BERULANG'].append(r)
        elif f_['n_hashtag'] >= 1 and len(kategori['HASHTAG']) < 3:
            kategori['HASHTAG'].append(r)
        elif f_['n_slang_replaced'] >= 2 and len(kategori['SLANG']) < 4:
            kategori['SLANG'].append(r)
        elif f_['n_url'] + f_['n_mention'] >= 1 and len(kategori['URL/MENTION']) < 3:
            kategori['URL/MENTION'].append(r)
    hasil = []
    for k, v in kategori.items():
        hasil.extend(v)
    return hasil[:jumlah]


def proses_dan_tampil():
    kamus = load_kamus(KAMUS)
    print('Kamus slang dimuat  :', len(kamus), 'entri')
    print()

    for split in ['train', 'val', 'test']:
        inp = os.path.join(DATA, split + '.jsonl')
        outp = os.path.join(DATA, split + '_preprocessed.jsonl')
        rows = load_jsonl(inp)
        hasil = []
        jitak_slang = 0
        for r in rows:
            teks_asli = r['text']
            fitur = hitung_fitur(teks_asli)
            teks_bersih, n_slang = bersihkan(teks_asli, kamus)
            jitak_slang += n_slang

            r_baru = dict(r)
            r_baru['text_clean'] = teks_bersih
            fitur['n_slang_replaced'] = n_slang
            r_baru['fitur'] = fitur
            hasil.append(r_baru)

        save_jsonl(outp, hasil)
        print('  {}: {} baris disimpan ke {}'.format(split, len(rows), outp))

    print()
    print('Total penggantian slang di seluruh data:', jitak_slang)

    # pembacaan sampel hanya di train agar ringkas
    print()
    print('=' * 70)
    print('SAMPEL SEBELUM / SESUDAH PREPROCESSING (train)')
    print('=' * 70)
    rows_train = load_jsonl(os.path.join(DATA, 'train_preprocessed.jsonl'))
    for r in pilih_sampel(rows_train):
        print()
        print('text_id  :', r['text_id'])
        print('SEBELUM  :', repr(r.get('text', '(tidak ada)')[:100]))
        print('SESUDAH  :', repr(r['text_clean'][:100]))
        print('fitur    : caps_ratio={}, n_allcaps={}, n_elongated={}, '
              'hashtag={}, mention={}, url={}, emoji={}, slang={}'
              .format(r['fitur']['caps_ratio'], r['fitur']['n_allcaps_word'],
                      r['fitur']['n_elongated'], r['fitur']['n_hashtag'],
                      r['fitur']['n_mention'], r['fitur']['n_url'],
                      r['fitur']['n_emoji'], r['fitur']['n_slang_replaced']))


# ------------------- EKSEKUSI -------------------
if __name__ == '__main__':
    proses_dan_tampil()
