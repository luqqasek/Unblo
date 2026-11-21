"""Reproducible paired analyses, uncertainty, and complete visual inspection artifacts."""
import argparse
import html
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from sklearn.metrics import average_precision_score
from scipy import ndimage as ndi
from common_moe import *
from metrics_moe import aggregate,mean

LABELS={'A0':'Universal / frozen M','Bglobal':'One calibrated threshold','A1':'Regime calibration only',
        'A2':'Small + M','A3':'Clean + M','A4':'S + M + L / soft gate','A5':'Uniform averaging',
        'A6':'Oracle regime (diagnostic)','Hard':'Hard predicted gate','S_only':'Small specialist only',
        'L_only':'Clean specialist only','U0':'Unweighted continuation','GateConstant':'Gate empty/full'}
ORDER=list(LABELS)


def table(rows,columns):
    def fmt(x):return f'{x:.4f}' if isinstance(x,(float,np.floating)) else '—' if x is None else str(x)
    return ['| '+' | '.join(label for key,label in columns)+' |','| '+' | '.join('---' for _ in columns)+' |']+[
        '| '+' | '.join(fmt(r.get(k)) for k,label in columns)+' |' for r in rows]


def load_runs():
    runs=[]
    for path in OUT.glob('*_fold*_seed*.json'):
        r=json.loads(path.read_text(encoding='utf-8'))
        if 'per_image' in r:runs.append(r)
    return sorted(runs,key=lambda r:(r['method'],ORDER.index(r['variant']),r['fold'],r['seed']))


def summarize(runs):
    rows=[dict(method=r['method'],variant=r['variant'],fold=r['fold'],seed=r['seed'],**x) for r in runs for x in r['per_image']]
    pd.DataFrame(rows).to_csv(OUT/'per_image_metrics.csv',index=False)
    folds=[dict(method=r['method'],variant=r['variant'],fold=r['fold'],seed=r['seed'],**r['metrics']) for r in runs]
    pd.DataFrame(folds).to_csv(OUT/'fold_metrics.csv',index=False)
    gates=json.loads((OUT/'screening.json').read_text());summary=[];seeds=[];strata=[]
    for method in METHODS:
        for variant in ORDER:
            rr=[r for r in runs if r['method']==method and r['variant']==variant]
            if not rr:continue
            pp=[x for r in rr for x in r['per_image']];full=len(rr)==9
            promoted=variant in ('A0','A1','Bglobal','GateConstant') or any(g['method']==method and g['variant']==variant and g['passed'] for g in gates)
            r=dict(method=method,variant=variant,protocol='3-fold x 3 seeds' if full else 'screen fold1 seed42',
                   role='candidate' if promoted else 'diagnostic/control',unique_images=len({x['name'] for x in pp}),
                   **aggregate(pp))
            r['miou_minus_constant']=r['miou']-r['constant_miou']
            for s in sorted({x['seed'] for x in rr}):
                z=[p for run in rr if run['seed']==s for p in run['per_image']]
                seeds.append(dict(method=method,variant=variant,seed=s,**aggregate(z)))
            for metric in ('miou','small_ap','small_topk','small_fg_iou','large_clean_iou','large_boundary_f1'):
                seed_values=[x[metric] for x in seeds if x['method']==method and x['variant']==variant]
                fold_values=[np.mean([x['metrics'][metric] for x in rr if x['fold']==f]) for f in sorted({x['fold'] for x in rr})]
                r[metric+'_seed_sd']=float(np.std(seed_values,ddof=1)) if full else None
                r[metric+'_fold_sd']=float(np.std(fold_values,ddof=1)) if full else None
            summary.append(r)
            for group,select in [('small',lambda x:x['area']<=.05),('medium',lambda x:.05<x['area']<=.5),
                    ('large',lambda x:x['area']>.5),('uncertain_gate',lambda x:x['gate_uncertain']),
                    ('confident_gate',lambda x:not x['gate_uncertain'])]+[(f'predicted_{label}',lambda x,i=i:x['predicted_regime']==i) for i,label in enumerate(('small','medium','large'))]:
                subset=[x for x in pp if select(x)]
                strata.append(dict(method=method,variant=variant,group=group,n=len(subset),unique_images=len({x['name'] for x in subset}),
                                   **{k:mean(subset,k) for k in ('miou','fg_iou','clean_iou','ap','auc','topk','boundary_f1','hole_clean_recall')}))
    pd.DataFrame(summary).to_csv(OUT/'comparison.csv',index=False);pd.DataFrame(seeds).to_csv(OUT/'seed_metrics.csv',index=False)
    pd.DataFrame(strata).to_csv(OUT/'stratified_metrics.csv',index=False)
    return rows,summary,seeds,strata


def paired(rows):
    lookup={(r['method'],r['variant'],r['name'],r['seed']):r for r in rows}
    pairs=[]
    for method in METHODS:
        variants=sorted({r['variant'] for r in rows if r['method']==method})
        for variant in variants:
            controls=['A0'] if variant!='A0' else []
            if variant in ('A3','A4'):controls+=['A1','A5','U0']
            for control in controls:
                records=[r for r in rows if r['method']==method and r['variant']==variant and (method,control,r['name'],r['seed']) in lookup]
                if not records:continue
                for scope,test,metrics in [
                    ('all',lambda r:True,['miou','ap','auc','fg_iou','clean_iou']),
                    ('folds2_3',lambda r:r['fold']!=1,['miou','ap','clean_iou']),
                    ('small',lambda r:r['area']<=.05,['ap','topk','fg_iou','missed','false_positive_area']),
                    ('large',lambda r:r['area']>.5,['clean_iou','clean_fpr','boundary_f1','hole_clean_recall'])]:
                    for metric in metrics:
                        by_image={};by_foldseed={}
                        for r in records:
                            if not test(r):continue
                            a=lookup[(method,control,r['name'],r['seed'])]
                            if r[metric] is None or a[metric] is None:continue
                            delta=r[metric]-a[metric]
                            by_image.setdefault(r['name'],[]).append(delta)
                            by_foldseed.setdefault((r['fold'],r['seed']),[]).append(delta)
                        if not by_image:continue
                        values=np.array([np.mean(v) for v in by_image.values()]);rng=np.random.default_rng(20261121)
                        lo,hi=np.quantile(values[rng.integers(len(values),size=(2000,len(values)))].mean(1),[.025,.975])
                        pairs.append(dict(method=method,variant=variant,control=control,scope=scope,metric=metric,
                            delta=float(values.mean()),ci_low=float(lo),ci_high=float(hi),unique_images=len(values),
                            improved_fold_seed_pairs=sum(np.mean(v)>0 for v in by_foldseed.values()),fold_seed_pairs=len(by_foldseed)))
    pd.DataFrame(pairs).to_csv(OUT/'paired_deltas.csv',index=False)
    return pairs


def plots(runs,rows,summary):
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    phase=json.loads((OUT/'phase_a.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,r in zip(axes,phase['methods']):
        cm=np.array(r['summary']['selected']['confusion']);ax.imshow(cm,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=['S','M','L'],yticklabels=['S','M','L'],
               xlabel='Predicted regime',ylabel='GT regime',title=r['method'])
    fig.tight_layout();fig.savefig(dest/'gate_confusion.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,method in zip(axes,METHODS):
        df=pd.read_csv(OUT/f'{method}_gate_per_image.csv');df=df[df.variant=='selected']
        ax.scatter(df.area,df.area_pred,c=df.gt_regime,cmap='viridis',s=14,alpha=.65)
        ax.plot([0,1],[0,1],'k--',lw=1);ax.set(xlabel='GT coverage',ylabel='OOF predicted coverage',title=method,xlim=(0,1),ylim=(0,1))
    fig.tight_layout();fig.savefig(dest/'coverage_regression.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(11,8))
    for i,method in enumerate(METHODS):
        df=pd.read_csv(OUT/f'{method}_gate_per_image.csv');df=df[df.variant=='selected']
        for j,col in enumerate(['p_small','p_medium','p_large']):
            for cls,label in enumerate(['GT small','GT medium','GT large']):
                axes[i,j].hist(df[df.gt_regime==cls][col],bins=np.linspace(0,1,11),histtype='step',label=label,lw=1.7)
            axes[i,j].set(title=method+' / '+col,xlabel='Probability',ylabel='Images')
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(dest/'gate_distributions.png',dpi=150);plt.close(fig)
    # Every panel uses a matched protocol: screen for PS/TM, complete OOF for SS.
    fig,axes=plt.subplots(3,3,figsize=(14,10))
    for i,method in enumerate(METHODS):
        full=any(r['method']==method and r['variant']=='A4' and r['protocol'].startswith('3-fold') for r in summary)
        rr=[r for r in rows if r['method']==method and (full or (r['fold']==1 and r['seed']==42))]
        for j,(regname,select) in enumerate([('small',lambda r:r['area']<=.05),('medium',lambda r:.05<r['area']<=.5),('large',lambda r:r['area']>.5)]):
            variants=['A0','A1','A2','A3','A4','A5','A6']
            vals=[np.mean([r['miou'] for r in rr if r['variant']==v and select(r)]) for v in variants]
            axes[i,j].bar(variants,vals,color=['#555','#aaa','#4d9','#59c','#e93','#89a','#cbb'])
            axes[i,j].set(ylim=(0,.7),ylabel='mIoU',title=f'{method}: {regname} / '+('OOF, 3 seeds' if full else 'screen only'))
    fig.tight_layout();fig.savefig(dest/'miou_by_regime.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    for i,method in enumerate(METHODS):
        full=any(r['method']==method and r['variant']=='A4' and r['protocol'].startswith('3-fold') for r in summary)
        rr=[r for r in rows if r['method']==method and (full or (r['fold']==1 and r['seed']==42))]
        for j,(key,select,label) in enumerate([('fg_iou',lambda r:r['area']<=.05,'small foreground IoU'),('clean_iou',lambda r:r['area']>.5,'large clean IoU')]):
            variants=['A0','A1','A2','A3','A4','A5','U0']
            vals=[np.mean([r[key] for r in rr if r['variant']==v and select(r)]) for v in variants]
            axes[i,j].bar(variants,vals);axes[i,j].set(ylabel=label,title=method+(' / OOF 3 seeds' if full else ' / screen only'))
    fig.tight_layout();fig.savefig(dest/'localization_by_regime.png',dpi=150);plt.close(fig)


def heat(p):return (matplotlib.colormaps['viridis'](p)[...,:3]*255).astype('uint8')


def overlay(rgb,gt,mask):
    im=np.asarray(rgb).astype(float)*.65
    im[gt]=.7*im[gt]+.3*np.array([0,255,0]);im[mask]=.6*im[mask]+.4*np.array([255,160,0])
    return im.clip(0,255).astype('uint8')


def gallery(runs,rows):
    dest=OUT/'gallery';dest.mkdir(exist_ok=True)
    bykey={(r['method'],r['variant'],r['fold'],r['seed']):r for r in runs}
    cases=[];panels=[]
    for p in sorted(OUT.glob('*_fold*_seed*_expert_maps.npz')):
        z=np.load(p);method=p.stem.split('_')[0];fold=int(p.stem.split('_fold')[1].split('_')[0]);seed=int(p.stem.split('_seed')[1].split('_')[0])
        a=bykey[(method,'A0',fold,seed)];b=bykey[(method,'A4',fold,seed)]
        s=bykey[(method,'S_only',fold,seed)];l=bykey[(method,'L_only',fold,seed)]
        final=np.load(OUT/f'{method}_A4_fold{fold}_seed{seed}.npz');base=np.load(OUT/f'{method}_A0_fold{fold}_seed{seed}.npz')
        clean_expert=np.load(OUT/f'{method}_L_only_fold{fold}_seed{seed}.npz')
        for i,name in enumerate(z['names']):
            gt=final['target'][i];rgb=Image.open(ROOT/'data/train/images'/f'{name}.png').convert('RGB').resize((128,128))
            out=final['probability'][i];mask=final['mask'][i];bm=base['mask'][i]
            err=np.zeros((128,128,3),np.uint8);err[mask&~gt]=[255,0,0];err[~mask&gt]=[0,90,255]
            images=[rgb,gt.astype('uint8')*255,heat(z['baseline'][i]),heat(z['small'][i]),heat(z['clean'][i]),heat(out),
                    overlay(rgb,gt,bm),overlay(rgb,gt,mask),err]
            titles=['RGB','GT watermark','M: P(watermark)','S: P(watermark)','L: P(clean)','A4: P(watermark)','GT green / M amber','GT green / A4 amber','FP red / FN blue']
            panel=Image.new('RGB',(9*132,185),'white');draw=ImageDraw.Draw(panel)
            for j,(im,title) in enumerate(zip(images,titles)):
                if not isinstance(im,Image.Image):im=Image.fromarray(im)
                panel.paste(im.convert('RGB'),(132*j,25));draw.text((132*j+2,8),title,fill='black')
            gate=z['gate'][i];r=b['per_image'][i];ar=a['per_image'][i]
            draw.text((2,158),f'{method} {name} | fold {fold} seed {seed} | GT {gt.mean():.1%} | gate S/M/L {gate[0]:.2f}/{gate[1]:.2f}/{gate[2]:.2f} | mIoU {ar["miou"]:.3f} -> {r["miou"]:.3f} | fixed [0,1] heatmaps',fill='black')
            filename=f'{method}_{name}'+(f'_seed{seed}' if seed!=42 else '')+'.jpg';panel.save(dest/filename,quality=88)
            panels.append(dict(method=method,name=str(name),file=filename,fold=fold,seed=seed,area=float(gt.mean())))
            sr=s['per_image'][i];lr=l['per_image'][i]
            holes=ndi.binary_fill_holes(gt)&~gt;labels,_=ndi.label(holes)
            sizes=np.bincount(labels.ravel());sizes[0]=0
            significant=np.isin(labels,np.flatnonzero(sizes>=16))
            hole_gain=(float((~clean_expert['mask'][i])[significant].mean())-float((~bm)[significant].mean())) if significant.any() else None
            cases.append(dict(method=method,name=str(name),file=filename,fold=fold,seed=seed,area=float(gt.mean()),
                delta_miou=r['miou']-ar['miou'],small_recovered=bool(gt.mean()<=.05 and ar['missed'] and not sr['missed']),
                baseline_missed=ar['missed'],small_missed=sr['missed'],moe_missed=r['missed'],
                significant_hole_pixels=int(significant.sum()),hole_recall_gain=hole_gain,
                wrong_confident_gate=bool(gate.max()>=.8 and gate.argmax()!=regime(gt[None])[0]),
                disagreement=float(np.std(np.stack([z['small'][i],z['baseline'][i],1-z['clean'][i]]),axis=0).mean())))
    pd.DataFrame(cases).to_csv(OUT/'visual_cases.csv',index=False)
    selections=[]
    for method in METHODS:
        candidates=[r for r in cases if r['method']==method]
        for category,subset,key,reverse in [
            ('small recovered',[r for r in candidates if r['small_recovered']],'delta_miou',True),
            ('clean hole recovered',[r for r in candidates if r['area']>.5 and r['hole_recall_gain'] is not None and r['hole_recall_gain']>0],'hole_recall_gain',True),
            ('wrong confident gate',[r for r in candidates if r['wrong_confident_gate']],'delta_miou',False),
            ('MoE worse',[r for r in candidates if r['delta_miou']<0],'delta_miou',False),
            ('expert disagreement',candidates,'disagreement',True)]:
            chosen=sorted(subset,key=lambda r:r[key],reverse=reverse)[:1]
            if chosen:
                row=dict(category=category,**chosen[0]);selections.append(row)
                Image.open(dest/row['file']).save(OUT/f'{method}_{category.replace(" ","_")}_example.png')
            else:selections.append(dict(method=method,category=category,note='No qualifying example in evaluated scope'))
    save(OUT/'visual_selections.json',selections)
    page=['<!doctype html><meta charset="utf-8"><title>Regime MoE: OOF</title><style>body{font:16px system-ui;background:#eee}img{width:100%;max-width:1188px}article{margin:20px 0}</style>',
          '<h1>Regime MoE — every evaluated expert image and seed</h1><p>PS/TM: screening fold only. StegaStamp: three seeds on the same images. No per-image color rescaling.</p>']
    for r in panels:page.append(f'<article><p>{r["method"]} {r["name"]} · fold {r["fold"]} · seed {r["seed"]} · GT {r["area"]:.1%}</p><img loading="lazy" src="{r["file"]}"></article>')
    (dest/'index.html').write_text('\n'.join(page),encoding='utf-8')
    return selections


def significant_holes():
    rows=[]
    for fold in (1,2,3):
        for seed in SEEDS:
            path=OUT/f'stegastamp_A3_fold{fold}_seed{seed}.npz'
            if not path.exists():continue
            a=np.load(OUT/f'stegastamp_A0_fold{fold}_seed{seed}.npz');b=np.load(path)
            for i,(name,g) in enumerate(zip(a['names'],a['target'])):
                if g.mean()<=.5:continue
                holes=ndi.binary_fill_holes(g)&~g;lab,_=ndi.label(holes)
                sizes=np.bincount(lab.ravel());sizes[0]=0;meaningful=np.isin(lab,np.flatnonzero(sizes>=16))
                if holes.any():rows.append(dict(name=str(name),fold=fold,seed=seed,all_holes=int(holes.sum()),
                    significant_holes=int(meaningful.sum()),base=float((~a['mask'][i])[meaningful].mean()) if meaningful.any() else None,
                    new=float((~b['mask'][i])[meaningful].mean()) if meaningful.any() else None))
    df=pd.DataFrame(rows);df['delta']=df['new']-df['base'];df.to_csv(OUT/'significant_hole_diagnostic.csv',index=False)
    return dict(unique_images=int(df[df.significant_holes>0].name.nunique()),baseline=float(df.base.mean()),candidate=float(df.new.mean()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--skip-visuals',action='store_true');args=ap.parse_args()
    runs=load_runs();rows,summary,seeds,strata=summarize(runs);deltas=paired(rows)
    gate=json.loads((OUT/'phase_a.json').read_text());screen=json.loads((OUT/'screening.json').read_text())
    # Additional threshold-free clean ranking diagnostic: a clean-mask gain must not
    # be attributed to newly learned spatial information solely from thresholded IoU.
    clean_rows=[]
    for r in runs:
        if r['method']!='stegastamp':continue
        z=np.load(OUT/f"{r['method']}_{r['variant']}_fold{r['fold']}_seed{r['seed']}.npz")
        for name,p,g in zip(z['names'],z['probability'],z['target']):
            if g.mean()<=.5:continue
            y=~g;p=1-p;k=int(y.sum());cut=np.partition(p.ravel(),p.size-k)[p.size-k]
            high=p>cut;eq=p==cut
            clean_rows.append(dict(variant=r['variant'],fold=r['fold'],seed=r['seed'],name=str(name),
                clean_ap=float(average_precision_score(y.ravel(),p.ravel())),
                clean_topk=float((y[high].sum()+(k-high.sum())*y[eq].mean())/k)))
    pd.DataFrame(clean_rows).to_csv(OUT/'large_clean_ranking_per_image.csv',index=False)
    clean_summary=pd.DataFrame(clean_rows).groupby('variant')[['clean_ap','clean_topk']].mean().reset_index().to_dict('records')
    save(OUT/'large_clean_ranking.json',clean_summary)
    hole_check=significant_holes();save(OUT/'significant_hole_summary.json',hole_check)
    gate_rows=[]
    for r in gate['methods']:
        for name,m in r['summary'].items():
            gate_rows.append(dict(method=r['method'],variant=name,**{k:m[k] for k in ('macro_f1','balanced_accuracy','log_loss','ece10','near_n','near_accuracy','near_confidence','uncertain_n','uncertain_accuracy')},
                small_precision=m['precision'][0],small_recall=m['recall'][0],large_precision=m['precision'][2],large_recall=m['recall'][2]))
    pd.DataFrame(gate_rows).to_csv(OUT/'gate_comparison.csv',index=False)
    complete=all((OUT/f'{m}_fold{f}_seed{s}_experts.json').exists() for m in METHODS
                 if any(g['method']==m and g['passed'] for g in screen) for f in (1,2,3) for s in SEEDS)
    data=dict(status='complete' if complete else 'partial',comparisons=summary,seeds=seeds,folds=[dict(method=r['method'],variant=r['variant'],fold=r['fold'],seed=r['seed'],**r['metrics']) for r in runs],
              paired=deltas,gate=gate_rows,screening=[{k:g[k] for k in ('method','variant','passed','small_pass','large_pass')} for g in screen])
    save(OUT/'summary.json',data)
    if not args.skip_visuals:plots(runs,rows,summary);gallery(runs,rows)
    lines=['# Regime-aware MoE: czy specjalizacja segmentacji ma sens?','',
           '**Status: '+data['status']+'.** Eksperyment real-only, bez testu konkursowego i leaderboardu.','']
    if (HERE/'CONCLUSIONS.md').exists():lines+=[(HERE/'CONCLUSIONS.md').read_text(encoding='utf-8'),'']
    lines+=['## Architektura i protokół','',
        'Zamrożony ekstraktor i jego cztery poziomy cech; StegaStamp z aktywną korektą inverse-STN. '
        'Wspólny trunk to cztery zamrożone projekcje 1×1 i fuzja wejść przy 32×32. '
        'M pozostaje dokładną zamrożoną głowicą bazową. S i L kopiują tylko jej decoder `fuse`; '
        'każdy ma 46 241 trenowane parametry. M nie jest ponownie uczony na outer train. '
        'Dodatkowy U0 to kontrola dalszego treningu identycznego dekodera bez wag reżimowych.','',
        'S: waga obrazu 4 dla GT≤5%, pozostałe 1; target watermark. L: waga 4 dla GT>50%, '
        'pozostałe 1; target jest prawdziwą maską clean. Ostatnia warstwa L startuje ze znakiem '
        'odwróconym względem M, więc początkowo 1−P(clean)=P(M). Clean Dice podkreśla rzadkie '
        'czyste piksele dużych masek; nie dodano osobnej wagi topologicznych dziur ani pseudoetykiet. '
        'S/L/U współdzielą zamrożone projekcje; nie tworzą trzech nowych backbone.','',
        'BCE+Dice 0.5/0.5, batch8, AdamW lr0.001, weight decay0.0001, maks.50 dodatkowych epok, '
        'patience8, wybór epoki przez zwykłe mean mIoU@0.5 na zapisanych inner validation. '
        'Wagi normalizowane sumą wag w batchu. PS/TM zachowują inner80/20; SS trzy inner foldy. '
        'Specjalistów wewnętrznych inicjalizowano z M nauczonego tylko na odpowiednim inner train. '
        'Outer specjaliści startują z właściwego niezmienionego outer M i uczą się wybraną liczbę epok. '
        'Seedy42/7/123, float32 CUDA bez TF32.','',
        'PS/TM seed42 wykorzystuje oryginalne checkpointy frozen_extractor; 7/123 zapisane modele '
        'A eksperymentu RGB/SRM o tej samej architekturze/procedurze. SS używa wszystkich trzech '
        'seedów aktywnego eksperymentu STN alignment. Nie wybierano najsłabszego seedu ani '
        'checkpointów synthetic→real. GPU odtworzyło istniejące mapy z tolerancją2e−5.','',
        'A4 = gS·S + gM·M + gL·(1−Lclean). A2 łączy S z M, A3 L z M, A5 średnia równomierna, '
        'Hard wybiera argmax gate. A6 używa GT reżimu wyłącznie jako diagnostyki; nie jest modelem '
        'deployable ani matematycznym maksimum po ekspertach. Wszystkie mieszanki mają próg '
        'wybrany na inner predictions z tej samej siatki0.3–0.7.','',
        '## Faza A — jakość gate','',
        '14 cech decoder-only: bit accuracy względem prawdziwej wiadomości, signed logits '
        '(średnia/SD/kwantyle), |logits|, log-likelihood wiadomości, oczekiwana poprawność bitu, entropia. '
        'Pooled wariant dodaje 8 składowych PCA z globalnych cech ekstraktora, dopasowanych wyłącznie '
        'na train danego dopasowania. StandardScaler również train-only. GBT nie potrzebuje '
        'skalowania. Gate nie wykorzystuje map segmentacji: nie ma wycieku z predykcji in-sample. '
        'Model wybiera najniższy inner log loss; nie zewnętrzne F1. Bez MLP i bez strojenia progów reżimów.','']
    lines+=['Gate jest dopasowany raz na outer fold i współdzielony między seedami głowic. '
            'Raportowana zmienność seedów dotyczy segmentacji; deterministycznych klasyfikatorów '
            'nie traktujemy jako trzech niezależnych powtórzeń.','']
    lines+=table(gate_rows,[('method','Metoda'),('variant','Gate'),('macro_f1','Macro F1'),('balanced_accuracy','Bal. acc.'),
        ('small_precision','S precision'),('small_recall','S recall'),('large_precision','L precision'),('large_recall','L recall'),('log_loss','Log loss'),('ece10','ECE10')])
    lines+=['','ECE10: top-label confidence, 10 równych binów. Near-boundary: odległość≤0.02 od5% '
            'lub≤0.10 od50%; uncertain: max gate<0.6. Te definicje ustalono przed oceną.','']
    lines+=table([r for r in gate_rows if r['variant']=='selected'],[('method','Metoda'),('near_n','N przy granicy'),
        ('near_accuracy','Accuracy przy granicy'),('near_confidence','Pewność przy granicy'),('uncertain_n','N niepewnych'),('uncertain_accuracy','Accuracy niepewnych')])
    lines+=['','Regresja area: Ridge alpha10, decoder-only, train-only scaler, predykcje ograniczone do[0,1].','']
    lines+=table([dict(method=r['method'],**r['area_regression']) for r in gate['methods']],[('method','Metoda'),('mae','Area MAE'),('spearman','Spearman')])
    lines+=['','Kalibracja regresji według binów area jest w phase_a.json; wszystkie prawdopodobieństwa '
            'i etykiety gate w *_gate_per_image.csv.','',
            '![Macierze pomyłek](results/figures/gate_confusion.png)','',
            '![Area regression](results/figures/coverage_regression.png)','',
            '![Rozkłady gate](results/figures/gate_distributions.png)','',
            '## Faza B — kalibracja bez nowych map','',
            'Bglobal dobiera jeden próg z pięciu. A1: sigmoid(logit(global_t)+g·offset), '
            'offset∈{−0.7,0,+0.7}³: 27 z góry określonych możliwości; zero preferowane przy remisie. '
            'Użyto wyłącznie inner held-out map M i cross-fitted gate. Outer GT nie wybiera progu. '
            'A1 zachowuje mapę probability, zmienia tylko próg per obraz: AP/AUC/Top-K muszą być identyczne z A0.','']
    cols=[('method','Metoda'),('variant','Wariant'),('miou','mIoU'),('small_ap','Small AP'),('small_topk','Small Top-K'),('small_fg_iou','Small FG IoU'),('large_clean_iou','Large clean IoU')]
    lines+=table([r for r in summary if r['variant'] in ('A0','Bglobal','A1','GateConstant')],cols)
    lines+=['','## Screening ekspertów i awans','',
            'Fold1/seed42. Awans wymaga poprawy obu small AP i Top-K (A2) lub obu large clean IoU '
            'i internal Boundary F1 (A3); A4 może spełnić którykolwiek warunek. Dodatkowo mIoU '
            'nie może spaść o więcej niż0.01. Zysk samego mIoU nie wystarcza.','']
    lines+=table(data['screening'],[('method','Metoda'),('variant','Wariant'),('passed','Awans'),('small_pass','Small'),('large_pass','Large')])
    lines+=table([dict(method=r['method'],variant=r['variant'],**r['metrics']) for r in runs if r['fold']==1 and r['seed']==42 and r['variant'] in ('A0','A1','A2','A3','A4','A5','A6','Hard','U0')],cols)
    lines+=['','W rodzinach z awansem trenowano te same dwa eksperckie dekodery dla pozostałych foldów/seedów. '
            'Pozostałe mieszanki tych samych map zachowano jako kontrole/diagnostykę, nie jako dodatkowo '
            'wybrane zwycięskie konfiguracje. Rodzin bez awansu nie rozszerzano.','',
            '## Pełne wyniki i kontrole','']
    lines+=table(summary,[('method','Metoda'),('variant','Wariant'),('protocol','Protokół'),('miou','mIoU'),('fg_iou','FG IoU'),
        ('clean_iou','Clean IoU'),('auc','AUC macro'),('ap','AP macro'),('miou_minus_constant','mIoU minus constant')])
    lines+=['','Screeningu57 obrazów nie należy porównywać z pełnym OOF170 obrazów. Pełny wynik jest '
            'średnią metryk po obrazach OOF i trzech seedach, a nie ensemble probabilities. '
            'Seedy powtarzają te same obrazy; to nadal170 unikalnych obserwacji/metodę.','',
            '## Lokalizacja małych watermarków','']
    lines+=table(summary,[('method','Metoda'),('variant','Wariant'),('protocol','Protokół'),('small_ap','AP'),('small_topk','Top-K'),
        ('small_fg_iou','FG IoU'),('small_empty_prediction','Puste maski'),('small_missed','Całkowicie pominięte'),('small_false_positive_area','FP / obraz')])
    lines+=['','Top-K zna prawdziwą liczbę pikseli GT: wyłącznie diagnostyka. Remisy są rozstrzygane przez '
            'oczekiwaną precyzję losowego wyboru wśród remisów. Całkowicie pominięty watermark oznacza TP=0; '
            'FP area oznacza liczbę fałszywie dodatnich pikseli podzieloną przez całą powierzchnię obrazu.','',
            '## Czyste regiony w dużych maskach','']
    lines+=table(summary,[('method','Metoda'),('variant','Wariant'),('protocol','Protokół'),('large_clean_iou','Clean IoU'),
        ('large_clean_fpr','FP / true clean'),('large_boundary_f1','Internal BF1@2'),('large_hole_clean_recall','Recall clean w dziurach')])
    lines+=['','Internal BF1 usuwa sztuczną granicę ramki obrazu; tolerancja2 komórki128×128. '
            'Dziura do diagnostyki to komponent clean całkowicie otoczony GT watermark, wykryty '
            'binary_fill_holes. To węższy zbiór niż wszystkie czyste regiony: pasy przy ramce '
            'nie są dziurami. Ta definicja nie tworzy żadnych etykiet do treningu. Hole recall '
            'nie wystarcza samodzielnie, ponieważ maska całkowicie clean uzyskałaby1. '
            'FP/true clean jest jednocześnie odsetkiem czystych pikseli błędnie oznaczonych jako watermark.','',
            'Dodatkowa diagnostyka StegaStamp: ranking pikseli clean przy GT watermark>50%, bez progu. '
            'Pozwala odróżnić lepsze maski binarne od lepszego uporządkowania czystych pikseli.','']
    lines+=table(clean_summary,[('variant','Wariant'),('clean_ap','Large clean AP'),('clean_topk','Large clean oracle Top-K')])
    lines+=['',f"Kontrola wrażliwości: po wyłączeniu komponentów dziur mniejszych niż16 komórek maski, "
            f"pozostają {hole_check['unique_images']} unikalne obrazy StegaStamp. Ich średni hole clean recall "
            f"A0→A3 wynosi {hole_check['baseline']:.4f}→{hole_check['candidate']:.4f}. To diagnostyka po treningu, "
            'nie filtr etykiet uczących ani kryterium awansu. W galerii przykłady odzyskanych dziur '
            'również wymagają komponentu≥16 komórek, aby nie prezentować przypadkowych pojedynczych pikseli.','',
            '## Zmienność i parowane różnice','']
    lines+=table([r for r in summary if r['protocol'].startswith('3-fold') and r['variant'] in ('A0','A1','A3','A4','A5','U0')],
        [('method','Metoda'),('variant','Wariant'),('miou_seed_sd','mIoU SD seedów'),('miou_fold_sd','mIoU SD foldów'),
         ('small_ap_seed_sd','Small AP SD seedów'),('large_clean_iou_seed_sd','Large clean SD seedów')])
    important=[r for r in deltas if r['variant'] in ('A3','A4') and r['method']=='stegastamp' and
        ((r['control']=='A0' and r['metric'] in ('miou','ap','topk','clean_iou','boundary_f1','hole_clean_recall')) or
         (r['control']!='A0' and r['scope']=='all' and r['metric']=='miou'))]
    lines+=table(important,[('variant','Wariant'),('control','Kontrola'),('scope','Zakres'),('metric','Metryka'),('delta','Delta'),('ci_low','CI95 low'),('ci_high','CI95 high')])
    lines+=['','Bootstrap2000 parowanych losowań obrazów, po uśrednieniu różnic seedów dla każdego obrazu. '
            'Przedziały są warunkowe, nie uwzględniają ponownego treningu ani wielokrotnego rozwoju na '
            'tych foldach. Foldy2/3 pokazano osobno od screeningowego foldu1. Wyniki warunkowane '
            'predicted regime i niepewnością gate, oraz medium precision/recall/area error: stratified_metrics.csv '
            'i comparison.csv. Wszystkie seedy i foldy: osobne CSV.','',
            '![mIoU według reżimu](results/figures/miou_by_regime.png)','',
            '![Lokalizacja według reżimu](results/figures/localization_by_regime.png)','',
            '## Przykłady, artefakty i reprodukcja','',
            '[Galeria wszystkich ocenionych obrazów i seedów](results/gallery/index.html), '
            '[dobór sukcesów i porażek](results/visual_selections.json), [porównania](results/comparison.csv), '
            '[per-image](results/per_image_metrics.csv), [foldy](results/fold_metrics.csv), '
            '[seedy](results/seed_metrics.csv), [parowane delty](results/paired_deltas.csv), '
            '[gate](results/gate_comparison.csv), [weryfikacja](results/verification.json).','',
            'Przykłady wybierane jawnie: small znaleziony przez S po miss M; największy wzrost hole recall '
            'w komponentach≥16 komórek; '
            'błędny gate z pewnością≥0.8; największa strata mIoU; największa średnia SD map ekspertów. '
            'Brak przykładu danej kategorii jest zapisany jako brak, nie zastąpiony wygodniejszym przypadkiem. '
            'Wszystkie mapy ekspertów, final probabilities, maski, progi, checkpointy i historie w results/.','',
            'Instrukcja uruchomienia: [README](README.md). Fazy A/B zostały zapisane przed treningiem '
            'ekspertów; SCREENING.md przed potwierdzeniem. Zachowano niezmienione wcześniejsze checkpointy '
            'i źródłowe splity. Nie wygenerowano danych syntetycznych, nie odczytywano testu konkursowego '
            'i nie wysłano zgłoszenia.','',
            '## Ograniczenia','',
            'Mały, wielokrotnie analizowany zbiór; brak nowego nietkniętego holdoutu. Gate ma mało '
            'danych medium, a oddzielenie medium/small jest trudne. S/L używają jednego ustalonego '
            'współczynnika4, bez szerokiego searchu. Zamrożone projekcje ograniczają zdolność specjalizacji. '
            'L zmienia orientację Dice na clean oraz wagę obrazów, więc nie izolujemy obu efektów. '
            'U0 kontroluje dalszy trening, ale nie wszystkie możliwe różnice capacity/ensemblingu. '
            'GT oracle A6 nie jest rozwiązaniem wdrożeniowym.','']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('REPORT',len(runs),'evaluated fold/seed/variant records',len(rows),'per-image records',data['status'],flush=True)


if __name__=='__main__':main()
