import json
import re
import glob
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# karakter yang mencurigakan: sirilik, cjk, replacement char, dsb
pola_mencurigakan = [
    (re.compile(r'[\u0400-\u04FF]'), 'sirilik (Cyrillic)'),
    (re.compile(r'[\u4E00-\u9FFF]'), 'huruf Mandarin'),
    (re.compile(r'\uFFFD'), 'replacement char (kotak tanya)'),
    (re.compile(r'[\u061F\u0640]'), 'karakter Arab aneh'),
    (re.compile(r'[\u01B7\u01BF]'), 'huruf langka latin'),
]

target_md = ['notebooks/step1_identifikasi_dataset.ipynb', 'notebooks/step2_analisis_agregasi.ipynb',
             'notebooks/step3_pipeline_final.ipynb', 'notebooks/step4_eda.ipynb',
             'notebooks/step5_eda_annotator_bias.ipynb']
target_md += glob.glob('reports/*.md') + ['README.md', 'config.yaml']
target_md += glob.glob('demo/*.py')

temuan = 0

for pass_file in target_md:
    try:
        if pass_file.endswith('.ipynb'):
            nb = json.load(open(pass_file, encoding='utf-8'))
            sumber_daftar = []
            for i, c in enumerate(nb['cells']):
                src = ''.join(c.get('source', []))
                sumber_daftar.append((f'cell {i} (id={c.get("id", "?")}, {c["cell_type"]})', src))
        else:
            with open(pass_file, encoding='utf-8') as f:
                isi = f.read()
            sumber_daftar = [('file', isi)]

        for lokasi, src in sumber_daftar:
            for pola, nama in pola_mencurigakan:
                for m in pola.finditer(src):
                    awal = max(0, m.start() - 30)
                    akhir = min(len(src), m.end() + 30)
                    potongan = src[awal:akhir].replace('\n', ' | ')
                    print(f'[{pass_file}] [{lokasi}] {nama}: ...{potongan}...')
                    temuan += 1
    except Exception as err:
        print(f'GAGAL {pass_file}: {err}')

print()
print('Total temuan karakter mencurigakan:', temuan)
