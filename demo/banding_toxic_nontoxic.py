# Perbandingan karakteristik teks toxic vs non-toxic
# Sumber data: file mentah indotoxic2024_annotated_data-3.jsonl

import json
import re
from collections import Counter

# Ganti path ini kalau foldernya beda
file_data = r'C:\SEMESTER 5\Project Sistem Cerdas\Projek\Dataset\indotoxic2024_annotated_data-3.jsonl'

barisan = []
with open(file_data, encoding='utf-8') as f:
    for line in f:
        barisan.append(json.loads(line))

# ambil 1 baris per teks biar teks yang sama (dinilai banyak anotator)
# tidak kehitung berulang
teks_aja = {}
for b in barisan:
    if b['text_id'] not in teks_aja:
        teks_aja[b['text_id']] = b
data = list(teks_aja.values())

toxic = [b for b in data if b['toxicity'] == 1]
nontoxic = [b for b in data if b['toxicity'] == 0]

print('Perbandingan teks toxic vs non-toxic (level teks unik)')
print('Total teks :', len(data))
print('Toxic      :', len(toxic))
print('Non-toxic  :', len(nontoxic))
print()

# pola emoji untuk deteksi (rentang karakter emoji unicode)
EMOJI = re.compile('[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]')


def ada_kapital3(t):
    return re.search(r'\b[A-Z]{3,}\b', t) is not None


def ada_huruf_ulang(t):
    return re.search(r'(.)\1{2,}', t) is not None


def ada_emoji(t):
    return EMOJI.search(t) is not None


def ada_mention(t):
    return re.search(r'@\w+', t) is not None


def ada_hashtag(t):
    return re.search(r'#\w+', t) is not None


def jml_kata(t):
    return len(t.split())


def rata(nilai):
    return round(sum(nilai) / len(nilai), 1)


def persen(nilai):
    return round(sum(nilai) / len(nilai) * 100, 1)


# bandingkan beberapa hal sederhana
print('Aspek                | toxic   | non-toxic')
print('-' * 45)

rata_toxic = rata([jml_kata(b['text']) for b in toxic])
rata_non = rata([jml_kata(b['text']) for b in nontoxic])
print(f'Rata-rata jumlah kata | {rata_toxic:5} | {rata_non:5}')

aspek = [
    ('pakai HURUF KAPITAL', ada_kapital3),
    ('pakai huruf berulang (anii)', ada_huruf_ulang),
    ('pakai emoji', ada_emoji),
    ('pakai mention @', ada_mention),
    ('pakai hashtag #', ada_hashtag),
]

for nama, cek in aspek:
    p_t = persen([1 if cek(b['text']) else 0 for b in toxic])
    p_n = persen([1 if cek(b['text']) else 0 for b in nontoxic])
    print(f'{nama:20s} | {p_t:4}% | {p_n:4}%')

# 5 kata paling sering di tiap kelompok (kata umum dibuang)
STOP = set('''yang dan di ke dari ini itu dengan untuk tidak ya aku saya kamu dia mereka kita kami
ada adalah akan sudah belum juga hanya saja lebih oleh pada atau jika kalau karena tapi tetapi
bisa mau gue gua gw lo lu udah dah banget bgt sih dong deh kok nih tuh emang memang saat
sama apa kenapa gimana kaya kayak begini begitu gitu nggak ngga gak ga
the a in of and is it to you my me'''.split())


def kata_tersering(kelompok):
    c = Counter()
    for b in kelompok:
        for w in re.findall(r"[a-zA-Z][a-zA-Z']+", b['text'].lower()):
            if w not in STOP and len(w) > 2:
                c[w] += 1
    return c


c_t = kata_tersering(toxic)
c_n = kata_tersering(nontoxic)
print()
print('5 kata tersering toxic     :', [w for w, _ in c_t.most_common(5)])
print('5 kata tersering non-toxic :', [w for w, _ in c_n.most_common(5)])

# kata yang cuma khas di toxic (jarang muncul di non-toxic)
print()
print('Kata khas toxic (hampir tidak ada di non-toxic):')
n_n = len(nontoxic)
khas = []
for w, n in c_t.most_common(40):
    p_n = c_n.get(w, 0) / n_n
    p_t = n / len(toxic)
    if p_t > 0.02 and p_n < p_t / 3:
        khas.append((w, round(p_t * 100, 1), round(p_n * 100, 1)))
for w, pt, pn in khas[:7]:
    print(f'  {w:12s} muncul di {pt}% teks toxic, cuma {pn}% di non-toxic')
