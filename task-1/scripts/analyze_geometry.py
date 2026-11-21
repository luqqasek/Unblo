"""Stage 0: ground-truth geometry on the official evaluation grid (no model inference)."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wmloc.utils import miou_image, downsample_mask


def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), **{k:float(v) for k,v in zip(['min','p10','p25','median','p75','p90','max'], np.quantile(a,[0,.1,.25,.5,.75,.9,1]))})


def analyze(data, out):
    out.mkdir(parents=True, exist_ok=True)
    meta_path = data / 'train/metadata.json'
    masks_path = data / 'train/masks_128.npz'
    meta = json.loads(meta_path.read_text())
    with np.load(masks_path) as z:
        masks = dict(zip(z['names'].tolist(), z['masks']))
    assert len(meta) == len(masks) == 510
    assert len({m['image_name'] for m in meta}) == 510
    assert {m['image_name'] for m in meta} == set(masks)
    rows = []
    for m in meta:
        name = m['image_name']
        g = masks[name]
        assert g.shape == (128,128) and np.isin(g, [0,1]).all()
        with Image.open(data / 'train/masks' / (name+'.png')) as im:
            full = (np.array(im)>127).astype(np.uint8)
        assert full.shape == (512,512)
        assert np.array_equal(downsample_mask(full), g)
        ys,xs = np.nonzero(g)
        bbox = np.zeros_like(g)
        if len(ys):
            y0,y1,x0,x1 = int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1
            bbox[y0:y1,x0:x1]=1
            bbox_cells=int(bbox.sum())
            fill=float(g.sum()/bbox_cells)
        else:
            y0=y1=x0=x1=None; bbox_cells=0; fill=None
        # Compare independent count-based formulas with official scoring.
        all_one_score=miou_image(np.ones_like(g),g)
        expected=1.0 if g.all() else float(g.mean()/2)
        assert np.isclose(all_one_score,expected)
        bbox_score=miou_image(bbox,g)
        if bbox_cells and not g.all():
            assert np.isclose(bbox_score, .5*(g.sum()/bbox_cells + (g.size-bbox_cells)/(g.size-g.sum())))
        rows.append(dict(image_name=name,method=m['method'],full_watermark=m.get('full_watermark'),
                         watermark_fraction_128=float(g.mean()),watermark_fraction_512=float(full.mean()),
                         bbox_fraction=bbox_cells/g.size,bbox_fill=fill,bbox_extra_cells=bbox_cells-int(g.sum()),
                         bbox_x0=x0,bbox_y0=y0,bbox_x1_exclusive=x1,bbox_y1_exclusive=y1,
                         miou_all_watermark=all_one_score,miou_gt_bbox=bbox_score))
    with (out/'per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    groups={'all':rows}
    for method in ['pixelseal','trustmark','stegastamp']:
        groups[method]=[r for r in rows if r['method']==method]
        for flag in [True,False]:
            groups[f'{method}/full_watermark={flag}']=[r for r in groups[method] if r['full_watermark']==flag]
    for key,items in groups.items():
        if not items:continue
        summary[key]={'n':len(items)}
        for col in ['miou_all_watermark','miou_gt_bbox','watermark_fraction_128','watermark_fraction_512','bbox_fill','bbox_fraction']:
            values=[r[col] for r in items if r[col] is not None]
            summary[key][col]=describe(values) if values else None
        scores=np.array([r['miou_gt_bbox'] for r in items])
        summary[key].update(bbox_perfect=int((scores==1).sum()),bbox_ge_095=int((scores>=.95).sum()),bbox_ge_099=int((scores>=.99).sum()),
                            empty_masks=sum(r['watermark_fraction_128']==0 for r in items),full_masks=sum(r['watermark_fraction_128']==1 for r in items))
    result={'sources':{str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [meta_path,masks_path]},
            'evaluation':'Official per-image two-class mIoU on 128x128 masks; axis-aligned bbox tightly enclosing all GT=1 cells. Not an achievable detector or optimal rectangle bound.',
            'groups':summary}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Etap 0 — geometria ground truth','',
           '510 masek treningowych. Wyniki mIoU i bboxy liczone na oficjalnej siatce 128 × 128. Każdy obraz ma równą wagę. Wszystkie maski 512 × 512 po oficjalnym downsamplingu zgadzają się z masks_128.npz.', '',
           'BBox to najmniejszy prostokąt osiowy obejmujący wszystkie jedynki GT, z włączonymi skrajnymi komórkami. Korzysta z ground truth: nie jest to wynik detektora ani maksimum po wszystkich prostokątach. Puste maski: predykcja bbox jest pusta.', '',
           '| Grupa | N | Cały obraz: mIoU | BBox GT: mIoU | Pokrycie 128: średnia | Wypełnienie bbox: średnia | BBox idealny | BBox ≥ 0,95 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,s in summary.items():
        lines.append(f"| {key} | {s['n']} | {s['miou_all_watermark']['mean']:.4f} | {s['miou_gt_bbox']['mean']:.4f} | {100*s['watermark_fraction_128']['mean']:.2f}% | {100*s['bbox_fill']['mean']:.2f}% | {s['bbox_perfect']} | {s['bbox_ge_095']} |")
    lines+=['','## Rozkłady — min / P10 / P25 / mediana / P75 / P90 / max','']
    for key in ['all','pixelseal','trustmark','stegastamp']:
        lines += [f'### {key}','']
        for metric in ['watermark_fraction_128','watermark_fraction_512','bbox_fill','miou_gt_bbox']:
            s=summary[key][metric]
            lines.append(f"- {metric}: "+' / '.join(f'{s[k]:.4f}' for k in ['min','p10','p25','median','p75','p90','max']))
    lines += ['', '## Definicje i odtwarzanie','',
              '- Pokrycie: liczba jedynek / liczba wszystkich pikseli lub komórek. Raportujemy osobno rozdzielczość 512 i siatkę oceny 128.',
              '- Wypełnienie bbox: liczba jedynek GT / powierzchnia bbox. Nie jest równe mIoU: mIoU uwzględnia też klasę czystą.',
              '- full_watermark=true: zgodnie z loaderem cały obraz był oznakowany, a część przywrócono z czystego oryginału. To nie jest etykieta konkretnego typu edycji.',
              '- Rozkłady obejmują pełne grupy, bez usuwania trudnych przykładów; brak wag po rozmiarze obszaru.',
              '- per_image.csv: wynik każdego obrazu i współrzędne bbox; summary.json: dokładne statystyki, liczby progowe i hashe źródeł.',
              '- Uruchom z task-1: `.venv/bin/python scripts/analyze_geometry.py` (Windows: `.venv\\Scripts\\python.exe scripts/analyze_geometry.py`).','']
    t = summary['trustmark']
    lines += ['', '## Interpretacja', '',
              f"TrustMark: bbox GT daje średnio {t['miou_gt_bbox']['mean']:.4f}; {t['bbox_perfect']}/170 masek odwzorowuje idealnie, {t['bbox_ge_095']}/170 osiąga co najmniej 0,95. Nie daje więc wyniku bliskiego ideałowi w całej grupie.",
              'Każda metoda ma po 85 przykładów full_watermark=true i 85 false. Rozkład pokrycia jest dwumodalny: małe oznakowane obszary i niemal całe obrazy z czystymi obszarami. Sama średnia około 50% ukrywa ten podział.',
              'Wysokie wypełnienie bbox nie oznacza wysokiego mIoU. Przykład: 90% obrazu ma watermark, a bbox obejmuje cały obraz. IoU watermarka = 0,9, IoU klasy czystej = 0, więc mIoU = 0,45.',
              'Wniosek: bbox dodatniego obszaru warto sprawdzić jako bazę dla części przykładów, ale trzeba osobno modelować czyste obszary i nieregularne kształty. Geometria GT nie mówi jeszcze, czy model potrafi te obszary znaleźć z obrazu. Wyniki train nie gwarantują takich samych wyników test.', '']
    (out/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,col,title in zip(axes,['watermark_fraction_128','bbox_fill','miou_gt_bbox'],['Watermarked fraction (128 grid)','GT bbox fill','GT bbox mIoU (two classes)']):
        for method,color in zip(['pixelseal','trustmark','stegastamp'],['#4056d9','#009e73','#d97706']):
            ax.hist([r[col] for r in groups[method]],bins=np.linspace(0,1,21),histtype='step',linewidth=2,label=method,color=color)
        ax.set(title=title,xlim=(0,1),xlabel='Fraction / score',ylabel='Images (n=170 per method)')
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle('Stage 0 · 510 training masks · official 128 × 128 evaluation grid')
    fig.savefig(out/'distributions.png',dpi=170)
    plt.close(fig)
    print('\n'.join(lines[:18]))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--out',type=Path,default=ROOT/'analysis/geometry')
    args=parser.parse_args();analyze(args.data,args.out)
