# Visualisasi perbandingan teks toxic vs non-toxic
# Pasangan dari banding_toxic_nontoxic.py
# Output: 1 gambar gabungan di folder_gambar

import json
import re
import os
from collections import Counter

import matplotlib.pyplot as plt

# Ganti path ini kalau foldernya beda
file_data = r'C:\SEMESTER 5\Project Sistem Cerdas\Projek\Dataset\indotoxic2024_annotated_data-3.jsonl'
folder_gambar = r'C:\SEMESTER 5\Project Sistem Cerdas\Projek\reports\figures'

os.makedirs(folder_gambar, exist_ok=True)

MERAH = '#c44e52'
BIRU = '#4c72b0'

barisan = []
with open(file_data, encoding='utf-8') as f:
    for line in f:
        barisan.append(json.loads(line))

# 1 baris per teks
teks_aja = {}
for b in barisan:
    if b['text_id'] not in teks_aja:
        teks_aja[b['text_id']] = b
data = list(teks_aja.values())

toxic = [b['text'] for b in data if b['toxicity'] == 1]
nontoxic = [b['text'] for b in data if b['toxicity'] == 0]

print('Toxic:', len(toxic), '| Non-toxic:', len(nontoxic))


# fungsi bantu
def rata(nilai):
    return sum(nilai) / len(nilai)


def persen(cek, kelompok):
    return sum(1 for t in kelompok if cek(t)) / len(kelompok) * 100


def jml_kata(t):
    return len(t.split())


EMOJI = re.compile('[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]')
ceks = {
    'HURUF KAPITAL\n(3 huruf atau lebih)': lambda t: re.search(r'\b[A-Z]{3,}\b', t) is not None,
    'huruf berulang\n(aniii)': lambda t: re.search(r'(.)\1{2,}', t) is not None,
    'emoji': lambda t: EMOJI.search(t) is not None,
    'mention\n@user': lambda t: re.search(r'@\w+', t) is not None,
    'hashtag\n#tag': lambda t: re.search(r'#\w+', t) is not None,
}

STOP = set('''yang dan di ke dari ini itu dengan untuk tidak ya aku saya kamu dia mereka kita kami
ada adalah akan sudah belum juga hanya saja lebih oleh pada atau jika kalau karena tapi tetapi
bisa mau gue gua gw lo lu udah dah banget bgt sih dong deh kok nih tuh emang memang saat
sama apa kenapa gimana kaya kayak begini begitu gitu nggak ngga gak ga
the a in of and is it to you my me'''.split())


def kata_tersering(kelompok):
    c = Counter()
    for t in kelompok:
        for w in re.findall(r"[a-zA-Z][a-zA-Z']+", t.lower()):
            if w not in STOP and len(w) > 2:
                c[w] += 1
    return c


c_t = kata_tersering(toxic)
c_n = kata_tersering(nontoxic)

# pilih kata yang khas toxic: sering di toxic, jarang di non-toxic
khas = []
for w, n in c_t.most_common(50):
    p_t = n / len(toxic)
    p_n = c_n.get(w, 0) / len(nontoxic)
    if p_t > 0.02 and p_n < p_t / 3:
        khas.append((w, p_t * 100, p_n * 100))

# gambar 2 x 2
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# panel 1: rata-rata panjang
axes[0, 0].bar(['Toxic', 'Non-toxic'],
               [rata([jml_kata(t) for t in toxic]), rata([jml_kata(t) for t in nontoxic])],
               color=[MERAH, BIRU])
axes[0, 0].set_title('Rata-rata panjang teks (kata)')
axes[0, 0].set_ylabel('jumlah kata')
for i, v in enumerate([rata([jml_kata(t) for t in toxic]), rata([jml_kata(t) for t in nontoxic])]):
    axes[0, 0].text(i, v + 0.5, str(round(v, 1)), ha='center', fontweight='bold')

# panel 2: fitur sosmed (bar berdampingan)
nama_aspek = list(ceks.keys())
p_t = [persen(ceks[n], toxic) for n in nama_aspek]
p_n = [persen(ceks[n], nontoxic) for n in nama_aspek]
x = range(len(nama_aspek))
lebar = 0.38
axes[0, 1].bar([i - lebar / 2 for i in x], p_t, lebar, label='Toxic', color=MERAH)
axes[0, 1].bar([i + lebar / 2 for i in x], p_n, lebar, label='Non-toxic', color=BIRU)
axes[0, 1].set_xticks(list(x))
axes[0, 1].set_xticklabels(nama_aspek, fontsize=8)
axes[0, 1].set_ylabel('% teks')
axes[0, 1].set_title('Fitur khas media sosial')
axes[0, 1].legend()

# panel 3: top 10 kata toxic vs non-toxic
t10 = c_t.most_common(10)
n10 = c_n.most_common(10)
y = range(10)
axes[1, 0].barh([i for i in y], [n for _, n in t10], color=MERAH, label='toxic')
axes[1, 0].set_yticks(y)
axes[1, 0].set_yticklabels([w for w, _ in t10][::-1])
axes[1, 0].set_title('Top 10 kata - TOXIC')
axes[1, 0].invert_yaxis()

# panel 4: kata khas toxic (jarang muncul di non-toxic)
khas_10 = khas[:10]
y2 = range(len(khas_10))
kata_k = [w for w, _, _ in khas_10]
p_t_k = [pt for _, pt, _ in khas_10]
p_n_k = [pn for _, _, pn in khas_10]
axes[1, 1].barh([i + 0.2 for i in y2], p_t_k, 0.4, label='di toxic', color=MERAH)
axes[1, 1].barh([i - 0.2 for i in y2], p_n_k, 0.4, label='di non-toxic', color=BIRU)
axes[1, 1].set_yticks(list(y2))
axes[1, 1].set_yticklabels(kata_k)
axes[1, 1].invert_yaxis()
axes[1, 1].set_xlabel('% teks yang memuat kata')
axes[1, 1].set_title('Kata khas TOXIC (hampir tidak ada di non-toxic)')
axes[1, 1].legend()

plt.suptitle('Perbandingan teks toxic vs non-toxic', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
nama_file = os.path.join(folder_gambar, '14_banding_toxic_nontoxic.png')
plt.savefig(nama_file, bbox_inches='tight', dpi=110)
print('Gambar disimpan:', nama_file)
plt.show()
