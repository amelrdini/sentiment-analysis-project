import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

dirp = 'data/processed'

print('== A. Jumlah baris aktual + flag, file {split}_preprocessed.jsonl (saat ini) ==')
for s in ['train', 'val', 'test']:
    p = os.path.join(dirp, s + '_preprocessed.jsonl')
    recs = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    flagged = sum(1 for r in recs if r.get('needs_manual_review') is True)
    false_n = sum(1 for r in recs if r.get('needs_manual_review') is False)
    print(f'{s}_preprocessed.jsonl : {len(recs):,} baris | needs_manual_review True: {flagged} | False: {false_n} | tanpa kolom: {len(recs) - flagged - false_n}')

print()
print('== B. Jumlah baris file {split}.jsonl (arsip, tanpa preprocessing) ==')
for s in ['train', 'val', 'test']:
    p = os.path.join(dirp, s + '.jsonl')
    n = sum(1 for l in open(p, encoding='utf-8') if l.strip())
    print(f'{s}.jsonl : {n:,} baris')

print()
print('== C. Konsistensi kolom split di file preprocessed ==')
for s in ['train', 'val', 'test']:
    p = os.path.join(dirp, s + '_preprocessed.jsonl')
    recs = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    salah = sum(1 for r in recs if r.get('split') != s)
    print(f'{s}_preprocessed : baris dgn kolom split != "{s}" : {salah}')

print()
print('== D. Cek ulang leakage group_id pada file PREPROCESSED ==')
grup = {}
for s in ['train', 'val', 'test']:
    p = os.path.join(dirp, s + '_preprocessed.jsonl')
    recs = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    grup[s] = set(r['group_id'] for r in recs)
print('grup di train DAN val  :', len(grup['train'] & grup['val']))
print('grup di train DAN test :', len(grup['train'] & grup['test']))
print('grup di val   DAN test :', len(grup['val'] & grup['test']))
print('total grup unik        :', len(grup['train'] | grup['val'] | grup['test']))

print()
print('== E. Audit step3 (data/interim/splits_summary.json) ==')
with open('data/interim/splits_summary.json', encoding='utf-8') as f:
    ss = json.load(f)
print('rows_input   :', ss['rows_input'])
print('units_final  :', ss['units_final'])
print('groups_final :', ss['groups_final'])

print()
print('== F. Rekap pemeriksaan jika angka yang dikutip adalah train 20.882 / val 2.610 / test 2.611 ==')
kutip = 20882 + 2610 + 2611
n_train = sum(1 for l in open(os.path.join(dirp, 'train_preprocessed.jsonl'), encoding='utf-8') if l.strip())
n_val = sum(1 for l in open(os.path.join(dirp, 'val_preprocessed.jsonl'), encoding='utf-8') if l.strip())
n_test = sum(1 for l in open(os.path.join(dirp, 'test_preprocessed.jsonl'), encoding='utf-8') if l.strip())
print('angka dikutip (jumlah)        :', f'{kutip:,}')
print('baris aktual di 3 file        :', f'{n_train + n_val + n_test:,}')
print('baris aktual per split        :', f'train {n_train:,} / val {n_val:,} / test {n_test:,}')
print('selisih dgn kutipan           :', f'{(n_train + n_val + n_test) - kutip:,} baris')
