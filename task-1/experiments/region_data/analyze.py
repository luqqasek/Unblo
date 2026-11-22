"""Report fixed-threshold paired generator comparisons; no threshold optimization."""
import json
from pathlib import Path
import numpy as np
from experiments.native_localizer.common import atomic_json
P=Path(__file__).resolve().parent;R=P/'results'

def summary(rows):
 groups={'all':rows,'fg_le1':[r for r in rows if 0<r['true_area']<=.01],'fg_1to5':[r for r in rows if .01<r['true_area']<=.05],'fg_gt5':[r for r in rows if .05<r['true_area']<1],'clean_le1':[r for r in rows if .99<=r['true_area']<1],'clean_1to5':[r for r in rows if .95<=r['true_area']<.99],'clean_minority':[r for r in rows if .5<r['true_area']<1]}
 return {k:dict(n=len(rs),**{metric:float(np.mean([r[metric] for r in rs if r.get(metric) is not None])) if any(r.get(metric) is not None for r in rs) else None for metric in ['miou','fg_iou','clean_iou','fg_ap','fp_area','complete_fg_miss','minority_iou']}) for k,rs in groups.items()}

def main():
 data={};lines=['# Dane obiektowe i małe pozostałości — dopasowane porównanie','',
 'Baza: message_localizer. Architektura, checkpoint startowy, budżet i loss identyczne; zmieniona receptura danych. Nie użyto testu konkursowego ani leaderboardu.','',
 '| Metoda | Etap | Dane | mIoU | FG IoU ≤1% | FG IoU 1–5% | FG IoU >5% | Clean IoU dziury ≤1% | Clean IoU dziury 1–5% |',
 '|---|---|---|---:|---:|---:|---:|---:|---:|']
 for family in ['pixelseal','trustmark']:
  data[family]={}
  for stage in ['synthetic','adapt','position']:
   pair={}
   variants=['objects_plus100','position_pairs100'] if stage=='position' else ['control','objects']
   for variant in variants:
    path=R/'runs'/family/(variant+'-adapt' if stage=='adapt' else variant)
    q=path/'real/per_image.json' if stage!='adapt' else path/'per_image.json'
    if not q.exists():continue
    rows=json.loads(q.read_text());s=summary(rows);pair[variant]=rows;data[family][stage+'_'+variant]=s
    vals=[s['all']['miou'],s['fg_le1']['fg_iou'],s['fg_1to5']['fg_iou'],s['fg_gt5']['fg_iou'],s['clean_le1']['clean_iou'],s['clean_1to5']['clean_iou']]
    lines.append(f'| {family} | {stage} | {variant} | '+' | '.join(f'{x:.5f}' if x is not None else 'n/a' for x in vals)+' |')
   if len(pair)==2:
    a={r['image_name']:r for r in pair[variants[0]]};b={r['image_name']:r for r in pair[variants[1]]};assert set(a)==set(b)
    delta=np.array([b[n]['miou']-a[n]['miou'] for n in a]);rng=np.random.default_rng(42);boot=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
    data[family][stage+'_paired']={'delta_miou':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'n':len(a)}
 lines+=['','## Różnice sparowane','']
 for f,stages in data.items():
  for k,v in stages.items():
   if k.endswith('_paired'):lines.append(f"- {f} / {k}: ΔmIoU {v['delta_miou']:+.5f}, bootstrap95% {v['ci95']}, n={v['n']}.")
 lines+=['','## Protokół i ograniczenia','',
 'Dwie kontynuacje po200kroków×32widoki od tego samego checkpointu400. LR5e-6/5e-5. Obiektowa gałąź:50% stare dane,25% same-content z maskami COCO,25% transplantacje/dodania między zdjęciami. Nowe osadzenia128train+32dev/metodę w kwadratach128/192/256/384; brak nowych źródeł, siła i wrapper bez zmian. Bez post-embedding JPEG/blur/resampling.','',
 'Adaptacja:150kroków/fold, batch24synthetic+8real, LR2e-6/2e-5,3oryginalne foldy, stały próg0,5. Obie gałęzie mają identyczną sekwencję realnych przykładów dzięki osobnemu RNG. Zapisane nowe dane odpowiadają całej recepturze, nie izolują wpływu samych masek od zmiany sampling weights i skali osadzania.','',
 'Drugi test:100 dodatkowych kroków od obiektowego checkpointu600, identyczne LR/loss/batch, RNG73. Porównuje dalszy stary sposób parowania z równopowierzchniowymi parami pozycyjnymi, skoncentrowanymi na małych regionach i dziurach. RGB nie jest przesuwane ani skalowane; zmienia się jedynie support na tym samym dokładnie oznakowanym źródle. Obie maski pary mają identyczną powierzchnię także po oficjalnym downsamplingu. To synthetic-only i adaptacyjny etap rozwojowy.', '',
 'Raport obejmuje rozwojowe OOF, wielokrotnie używane do badań. Bootstrap jest warunkowy po obrazach i nie uwzględnia ponownego treningu/całej zależności źródeł. Wynik na0 lub1 obrazie w małej grupie nie jest stabilnym dowodem. Nie zmieniono produkcyjnego#2163.','',
 'Artefakty JURECA: `/p/project1/training2666/pszenny1/native-localizer/region-runs/`; dokładne dodatkowe osadzenia: `region-data/`. Konfiguracje, realizowane proporcje, provenance i geometria/sygnał w `results/runs/`. Kod: prepare.py, data.py, experiment.py, adapt.py. Wszystkie etykiety pochodzą z pochodzenia pikseli, nigdy z niezerowego residualu.','',
 'Reprodukcja w przydziale4GPU: `python -m experiments.region_data.job`, potem `python -m experiments.region_data.adapt_job`. Lokalna analiza po pobraniu JSON: `PYTHONPATH=task-1:task-1/src task-1/.venv/bin/python -m experiments.region_data.analyze`.']
 audit_path=R/'audit_summary.json'
 if audit_path.exists():
  audit=json.loads(audit_path.read_text())
  lines+=['','## Geometria i sygnał — diagnostyka','', '| Metoda | Źródło | n partial | Wypełnienie bbox | Komponenty | Mediana szerokości, średnia po obrazach (komórki128) |', '|---|---|---:|---:|---:|---:|']
  for family,groups in audit.items():
   for name in ['real','control','objects']:
    g=groups[name];lines.append(f"| {family} | {name} | {g['n']} | {g['bbox_fill']['mean']:.4f} | {g['components']['mean']:.2f} | {g['width_p50_grid']['mean']:.2f} |")
  lines+=['','Próbka syntetyczna:128 primary/dev/gałąź, niezależna od stałych64 przykładów oceny modelu. Komponenty:4-connectivity. Nowe dane przybliżają bbox fill, ale nie usuwają rozbieżności liczby komponentów i szerokości. To opis rozkładów, nie dowód przyczyny wyniku.','', '| Metoda | Reżim osadzenia | n dev | RMS | Udział niezmienionych kanałów |','|---|---|---:|---:|---:|']
  for family,groups in audit.items():
   for regime,g in groups['encoding_regimes'].items():lines.append(f"| {family} | {regime} | {g['n']} | {g['rms']['mean']:.4f} | {g['zero_fraction']['mean']:.4f} |")
  lines+=['','RMS w jednostkach uint8, bez wzmacniania. Udział zerowych kanałów różni się od udziału całych niezmienionych pikseli; oba zapisano oddzielnie w audit_summary.json, gdzie dostępne. Rzeczywistych clean originals nie mamy, więc nie porównujemy tych residuali bezpośrednio z realnymi. Globalne bit accuracy małych fragmentów jest bliskie0,5 mimo niezerowego residualu; takie przykłady pozostają w treningu.']
 atomic_json(R/'summary.json',data);(P/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({f:{k:v for k,v in d.items() if k.endswith('_paired')} for f,d in data.items()},indent=2))
if __name__=='__main__':main()
