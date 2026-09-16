# Script sederhana untuk membuktikan jumlah toxic vs non-toxic
# Sumber data: file mentah indotoxic2024_annotated_data-3.jsonl (tanpa olahan)

import json

# Ganti path ini kalau foldernya beda
file_data = r'C:\SEMESTER 5\Project Sistem Cerdas\Projek\Dataset\indotoxic2024_annotated_data-3.jsonl'

# baca file jsonl
barisan = []
with open(file_data, encoding='utf-8') as f:
    for line in f:
        barisan.append(json.loads(line))

print('Total baris terbaca :', len(barisan))

# hitung nilai kolom toxicity
toxic = 0
nontoxic = 0
nilai_aneh = []

for baris in barisan:
    if baris['toxicity'] == 1:
        toxic += 1
    elif baris['toxicity'] == 0:
        nontoxic += 1
    else:
        # kalau ada nilai selain 0 atau 1 akan masuk sini
        nilai_aneh.append(baris['toxicity'])

print()
print('Jumlah non-toxic (toxicity=0):', nontoxic)
print('Jumlah toxic     (toxicity=1):', toxic)
print('Nilai lain selain 0/1       :', len(nilai_aneh))

# persentase
total = len(barisan)
print()
print('Persentase non-toxic :', round(nontoxic / total * 100, 2), '%')
print('Persentase toxic     :', round(toxic / total * 100, 2), '%')

# perbandingan
print()
print('Perbandingan non-toxic vs toxic =', round(nontoxic / toxic, 1), ': 1')
print()
if nontoxic > toxic:
    print('KESIMPULAN: data non-toxic lebih banyak dari toxic, terbukti.')
else:
    print('KESIMPULAN: data toxic lebih banyak dari non-toxic.')

# bandingkan juga dengan file hasil olahan step3 kalau ada
import os
file_train = r'C:\SEMESTER 5\Project Sistem Cerdas\Projek\data\processed\train.jsonl'
if os.path.exists(file_train):
    print()
    print('=' * 50)
    print('Bonus: cek juga file hasil olahan step3')
    for nama in ['train', 'val', 'test']:
        t = 0
        nt = 0
        with open(r'C:\SEMESTER 5\Project Sistem Cerdas\Projek\data\processed\\' + nama + '.jsonl', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r['toxicity'] == 1:
                    t += 1
                else:
                    nt += 1
        print(f'{nama:6s}: non-toxic {nt}, toxic {t}, total {nt + t}')
