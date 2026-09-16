# Verifikasi final hasil split step3
# Bukti 1: tidak ada group_id yang muncul di lebih dari satu split
# Bukti 2: distribusi 6 label per split (proporsi terjaga)
# Kocekannya langsung ke file train.jsonl / val.jsonl / test.jsonl

import json

# ganti path ini kalau foldernya beda
folder = r'C:\SEMESTER 5\Project Sistem Cerdas\Projek\data\processed'

LABEL_COLS = ['toxicity', 'profanity_obscenity', 'threat_incitement_to_violence',
              'insults', 'identity_attack', 'sexually_explicit']

# baca ketiga file
data = {}
for nama in ['train', 'val', 'test']:
    baris = []
    with open(folder + '\\' + nama + '.jsonl', encoding='utf-8') as f:
        for line in f:
            baris.append(json.loads(line))
    data[nama] = baris

print('Ukuran file:')
for nama in ['train', 'val', 'test']:
    print(' ', nama, ':', len(data[nama]), 'baris')

# kumpulkan group_id per split, sekalian cek nilai kolom split konsisten
grup = {}
for nama in data:
    s = set()
    salah_split = 0
    for r in data[nama]:
        s.add(r['group_id'])
        if r['split'] != nama:
            salah_split += 1
    grup[nama] = s
    print(f'  cek kolom split di {nama}: {salah_split} baris tidak cocok (harus 0)')

print()
print('BUKTI 1: group_id lintas split')
irisan_1 = grup['train'] & grup['val']
irisan_2 = grup['train'] & grup['test']
irisan_2 = grup['val'] & grup['test']
print('  grup di train DAN val  :', len(irisan_1))
print('  grup di train DAN test :', len(irisan_2))
print('  grup di val   DAN test :', len(irisan_2))
total_grup = len(grup['train'] | grup['val'] | grup['test'])
print('  total grup unik        :', total_grup)
if len(irisan_1) == 0 and len(irisan_2) == 0 and len(irisan_2) == 0:
    print('  => TIDAK ADA satupun group_id yang muncul di lebih dari satu split. Terbukti.')
else:
    print('  => ADA KEBOCORAN, harus ditinjau ulang!')

print()
print('BUKTI 2: distribusi 6 label per split')
print()
print(f'{"label":30s} {"split":7s} {"positif":8s} {"dari":8s} {"%":7s}')
print('-' * 58)
for c in LABEL_COLS:
    for nama in ['train', 'val', 'test']:
        rows = data[nama]
        n = sum(r[c] for r in rows)
        print(f'{c:30s} {nama:7s} {n:8d} {len(rows):8d} {n/len(rows)*100:6.2f}%')
    print()

# detail sexually_explicit
total_sexual = sum(r['sexually_explicit'] for r in data['train'] + data['val'] + data['test'])
print('Catatan sexually_explicit:')
print('  total positif di seluruh split:', total_sexual)
print('  val + test cuma punya belasan contoh positif, wajar karena')
print('  labelnya paling langka (0,5%) dan grup tidak boleh terpecah.')

# cocokkan dengan splits_summary.json
print()
print('Cek silang dengan data/interim/splits_summary.json')
with open(r'C:\SEMESTER 5\Project Sistem Cerdas\Projek\data\interim\splits_summary.json', encoding='utf-8') as f:
    s = json.load(f)
cocok = True
for c in LABEL_COLS:
    for nama in ['train', 'val', 'test']:
        di_file = sum(r[c] for r in data[nama]) / len(data[nama])
        di_summary = s['label_distribution_unit_level'][c][nama]
        if abs(di_file - di_summary) > 0.000001:
            cocok = False
            print('  BEDA:', c, nama)
print('Semua proporsi cocok dengan splits_summary.json:', cocok)
