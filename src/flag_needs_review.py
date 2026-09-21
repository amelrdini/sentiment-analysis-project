import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

processed_dir = r'C:\SEMESTER 5\Project Sistem Cerdas\Projek\data\processed'
splits = ['train', 'val', 'test']

for split in splits:
    path = processed_dir + '\\' + split + '_preprocessed.jsonl'
    with open(path, encoding='utf-8') as f:
        recs = [json.loads(l) for l in f if l.strip()]

    n_flagged = 0
    n_hati_cina = 0
    for r in recs:
        n = int(r['fitur']['n_slang_replaced'])
        r['needs_manual_review'] = (n >= 50)
        if r['needs_manual_review']:
            n_flagged += 1
        if 'hati cina' in r.get('text_clean', ''):
            n_hati_cina += 1

    with open(path, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print('split {}: {} teks, {} ditandai needs_manual_review (>= 50), {} teks masih memuat frasa "hati cina"'.format(
        split, len(recs), n_flagged, n_hati_cina))
