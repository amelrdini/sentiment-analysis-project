# Cari kalimat mirip "orang palestina dibunuh oleh israel" di dataset,
# lalu lihat anotator menilainya apa

import json

file_data = r'C:\SEMESTER 5\Project Sistem Cerdas\Projek\Dataset\indotoxic2024_annotated_data-3.jsonl'

barisan = []
with open(file_data, encoding='utf-8') as f:
    for line in f:
        barisan.append(json.loads(line))

# ambil 1 baris per teks
teks_aja = {}
for b in barisan:
    if b['text_id'] not in teks_aja:
        teks_aja[b['text_id']] = b

# cari teks yang memuat kata kunci
kunci = ['dibunuh']
hasil = []
for b in teks_aja.values():
    t = b['text'].lower()
    if 'dibunuh' in t and ('palestina' in t or 'gaza' in t or 'israel' in t):
        hasil.append(b)

print('Jumlah teks yang memuat "dibunuh" + palestina/gaza/israel :', len(hasil))
print()
toxic = 0
nontoxic = 0
for b in hasil[:12]:
    print('toxicity =', b['toxicity'], '|', repr(b['text'][:85]))
    if b['toxicity'] == 1:
        toxic += 1
    else:
        nontoxic += 1
print()
print('Dari sampel di atas: non-toxic', nontoxic, '| toxic', toxic)

# hitung semua
t_semua = sum(1 for b in hasil if b['toxicity'] == 1)
print()
print('Semua', len(hasil), 'teks: non-toxic', len(hasil) - t_semua, '| toxic', t_semua)
