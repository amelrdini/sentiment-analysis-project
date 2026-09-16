# Script untuk mengecek kalimat duplikat dalam dataset
# Sumber data: file mentah indotoxic2024_annotated_data-3.jsonl

import json
import csv
from collections import defaultdict

# Ganti path ini kalau foldernya beda
file_data = r'C:\SEMESTER 5\Project Sistem Cerdas\Projek\Dataset\indotoxic2024_annotated_data-3.jsonl'

barisan = []
with open(file_data, encoding='utf-8') as f:
    for line in f:
        barisan.append(json.loads(line))

print('Total baris anotasi :', len(barisan))


# samakan dulu bentuk kalimat: spasi berlebih dibuang, huruf kecil semua
# jadi "Halo  dunia" dan "halo dunia" dianggap kalimat yang sama
def normal(text):
    return ' '.join(text.split()).lower()


# kelompokkan baris berdasarkan kalimat yang sama
kelompok = defaultdict(list)
for i, b in enumerate(barisan):
    kelompok[normal(b['text'])].append(i)

print('Jumlah kalimat unik (setelah disamakan bentuknya):', len(kelompok))
print()

# pilah: kelompok normal vs duplikat lintas text_id
normal_aja = 0          # 1 text_id saja = wajar (beda anotator menilai teks sama)
duplikat_lintas = 0     # kalimat sama muncul di text_id berbeda = duplikat bermasalah
baris_lintas = 0

daftar_duplikat = []    # (kalimat, daftar index baris)

for kalimat, idxs in kelompok.items():
    text_ids = set(barisan[i]['text_id'] for i in idxs)
    if len(text_ids) > 1:
        duplikat_lintas += 1
        baris_lintas += len(idxs)
        daftar_duplikat.append((kalimat, idxs))
    else:
        normal_aja += 1

print('A. Kalimat yang hanya muncul di 1 text_id :', normal_aja, '(wajar, itu beda anotator)')
print('B. Kalimat yang muncul di >1 text_id      :', duplikat_lintas, '(duplikat bermasalah)')
print('   Baris yang terlibat duplikat B         :', baris_lintas, 'dari', len(barisan))
print()

# 5 kalimat yang paling sering muncul dobel
daftar_duplikat.sort(key=lambda g: len(g[1]), reverse=True)
print('5 kalimat yang paling banyak muncul dobel:')
for kalimat, idxs in daftar_duplikat[:5]:
    text_ids = sorted(set(barisan[i]['text_id'] for i in idxs))
    print('-', len(idxs), 'kali, di text_id:', ', '.join(text_ids))
    print('  ', repr(kalimat[:70]))
print()

# contoh detail satu kelompok duplikat
print('Contoh detail satu kelompok duplikat (semua barisnya):')
kalimat, idxs = daftar_duplikat[0]
print('Kalimat:', repr(kalimat[:80]))
for i in idxs:
    b = barisan[i]
    print('   text_id', b['text_id'],
          '| annotator', b['annotator_id'],
          '| toxicity', b['toxicity'])
print()

# simpan daftar lengkap duplikat ke CSV, bisa dibuka di excel
with open('daftar_duplikat.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['kalimat', 'jumlah_kemunculan', 'text_id', 'annotator_id', 'toxicity'])
    for kalimat, idxs in daftar_duplikat:
        for i in idxs:
            b = barisan[i]
            w.writerow([b['text'], len(idxs), b['text_id'], b['annotator_id'], b['toxicity']])

print('Daftar lengkap semua kalimat duplikat disimpan ke: daftar_duplikat.csv')
print()
print('Catatan: duplikat tipe B inilah alasan step3 membuat group_id,')
print('supaya kalimat yang sama tidak terpecah ke train dan test sekaligus.')
