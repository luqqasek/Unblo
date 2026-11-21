"""Append measured E1/E2 evidence, update conclusions and create fusion galleries."""
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from report import table
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss

def main():
    out=OUT/'fusion';summary=json.loads((out/'summary.json').read_text())
    df=pd.read_csv(out/'per_image_metrics.csv');metrics=pd.DataFrame(summary['metrics']);comps=pd.DataFrame(summary['comparisons'])
    records,b,g,_,_=cohort();idx={r['image_id']:i for i,r in enumerate(records)}
    from fusion import labels
    orientation_rows=[];shift_rows=[]
    forensic=np.stack([np.load(OUT/'detectivesam'/f'{r["image_id"]}.npz')['probability'] for r in records])
    for family in METHODS:
        for fold in (1,2,3):
            z=np.load(out/f'{family}_fold{fold}_E2_forensic.npz');ns=z['names'].tolist();ids=np.array([idx[n] for n in ns])
            truth=labels(b[ids],forensic[ids],g[ids],False);probs=z['gate_probability'];pred=probs.argmax(1)
            inner=np.load(out/f'{family}_fold{fold}_inputs.npz');ri=np.array([idx[n] for n in inner['refit_names']]);ci=np.array([idx[n] for n in inner['cal_names']])
            innerlabels=labels(inner['refit_probability'],forensic[ri],g[ri],False);majority=int(np.bincount(innerlabels,minlength=3).argmax())
            shift_rows.append(dict(family=family,fold=fold,cal_w_mean=float(inner['cal_probability'].mean()),
                                   refit_w_mean=float(inner['refit_probability'].mean()),outer_w_mean=float(b[ids].mean()),
                                   cal_gt_area=float(g[ci].mean()),refit_gt_area=float(g[ri].mean()),outer_gt_area=float(g[ids].mean())))
            for j,n in enumerate(ns):orientation_rows.append(dict(image_id=n,family=family,fold=fold,target=int(truth[j]),predicted=int(pred[j]),majority=majority,
                                                               p_uninformative=float(probs[j,0]),p_direct=float(probs[j,1]),p_inverse=float(probs[j,2])))
    od=pd.DataFrame(orientation_rows);od.to_csv(out/'orientation_per_image.csv',index=False);pd.DataFrame(shift_rows).to_csv(out/'map_distribution_shift.csv',index=False)
    orientation=[]
    for family in ['all',*METHODS]:
        d=od if family=='all' else od[od.family==family];informative=d.target>0
        orientation.append(dict(family=family,n=len(d),accuracy=float(accuracy_score(d.target,d.predicted)),
                                balanced_accuracy=float(balanced_accuracy_score(d.target,d.predicted)),
                                majority_accuracy=float(accuracy_score(d.target,d.majority)),informative_n=int(informative.sum()),
                                informative_orientation_accuracy=float(((d.loc[informative,'p_direct']>=d.loc[informative,'p_inverse']).astype(int).replace({0:2})==d.loc[informative,'target']).mean()) if informative.any() else np.nan,
                                log_loss=float(log_loss(d.target,d[['p_uninformative','p_direct','p_inverse']],labels=[0,1,2])),
                                confusion=confusion_matrix(d.target,d.predicted,labels=[0,1,2]).tolist()))
    save(out/'orientation_summary.json',orientation)
    lines=['## Faza E: uczona fuzja Detective SAM','',
           'Fazę E uruchomiono po zakończeniu C/D: jedynie Detective SAM przeszedł bramkę. Powstały63 małe głowice baseline do generowania poprawnych wewnętrznych map. Ekstraktory watermarku i wszystkie modele forensic pozostały zamrożone. StegaStamp zachował inverse-STN i inicjalizację istniejącym checkpointem pretrain_seed42; nie wytwarzano ani nie czytano nowych syntetyków.',
           '', 'Każdy outer-fold ma osobny meta-train i meta-validation zgodny z istniejącym inner_train/inner_validation. Mapy meta-train są3-way cross-fitted **wyłącznie w meta-train**; model generujący mapy meta-validation też nigdy go nie widział. Po wyborze parametrów fuzja jest dopasowana ponownie na oddzielnych3-way cross-fitted mapach całego outer-train. Każda głowica wybiera epokę na swoim własnym80/20 podziale, potem uczy się od nowa. Żadna mapa fitting/calibration nie ma producenta uczonego lub wybieranego na outer-validation. [Audyt63 modeli](results/verification.json).',
           '', 'Mapy wewnętrzne pochodzą z mniejszych zbiorów niż docelowa głowica. Wewnętrzna selekcja głowicy używa prostego80/20 także dla StegaStamp, podczas gdy zapisany zewnętrzny baseline używał stabilnej selekcji3-fold. To ograniczenie dopasowania rozkładu meta-danych; kontrola i forensic dostają dokładnie te same mapy. Zewnętrzne predykcje baseline są oryginalnymi zachowanymi OOF.',
           '', '**E1:** sigmoid(b+w1·logit(W)+w2·logit(F)), clipping1e−5. Kontrola ma tyle samo3 parametrów: sigmoid(b+w1·logit(W)+w2·logit(W)²/5). Strata BCE na1024 deterministycznie próbkowanych pikselach na obraz; równa waga obrazów; kara L2 {0,001;0,01;0,1} względem tożsamości. Dobór kary i progu0,05…0,95 wyłącznie na meta-validation.',
           '', '**E2:** miękki multinomial logistic gate z48 nominalnymi parametrami: po7 statystyk W/F i globalne bit accuracy znanej wiadomości. Statystyki: mean/std/kwantyle10,50,90/area@0,5/entropy. Klasy: zachowaj W, użyj F, użyj1−F. Etykiety treningowe wyznacza przewaga odpowiedniej akcji nad W>0,02 mIoU. Finalna mapa jest miękką mieszanką z osłabieniem α∈{0,25;0,5;1}. C∈{0,01;0,1;1} i próg wybiera meta-validation. Kontrola o tym samym rozmiarze używa wyłącznie statystyk W, ich kwadratów i bit accuracy; miesza W/pustą/pełną maskę. To kontrola zdolności kalibracji i przewidywania dominującej klasy.',
           '', 'Wyniki E2 nie wykorzystują per-image GT do wyboru kierunku przy inferencji. GT jest wyłącznie celem uczenia gate na dopuszczonym treningu. Nie używamy full_watermark jako wejścia, bo ta informacja jest treningowym metadata.', '']
    lines+=table(metrics[(metrics.family=='all')&(metrics.stratum=='all')],{'Wariant':'variant','mIoU':'miou','FG IoU':'fg_iou','Clean IoU':'clean_iou','AUC':'auc','AP':'ap','BF1':'boundary_f1'})
    lines+=['','### Porównania parowane','']
    lines+=table(comps,{'Rodzina':'family','Wariant':'variant','Kontrola':'control','Delta mIoU':'delta','CI95':'ci95','Dodatnie obrazy':'improved','Ujemne obrazy':'damaged'})
    lines+=['','### Czy gate rozpoznaje interpretację bez GT?','',
            'Klasy0/1/2: zachowaj baseline / direct / inverse; target przyznaje direct/inverse tylko przy przewadze>0,02 nad baseline przy progu0,5. Majority control wybiera najczęstszą klasę z odpowiedniego outer-train. Accuracy może być wysokie przez częste „zachowaj baseline”; balanced accuracy i dokładność kierunku wśród informatywnych przypadków są osobnymi diagnostykami. Wszystkie targety zewnętrzne użyto tylko do tej oceny.', '']
    lines+=table(pd.DataFrame(orientation),{'Rodzina':'family','Accuracy':'accuracy','Balanced accuracy':'balanced_accuracy','Majority accuracy':'majority_accuracy','Informative N':'informative_n','Kierunek w informative':'informative_orientation_accuracy'})
    lines+=['','[Macierze pomyłek i log-loss](results/fusion/orientation_summary.json), [per-image gate](results/fusion/orientation_per_image.csv), [rozkłady map wewnętrznych i zewnętrznych](results/fusion/map_distribution_shift.csv).']
    for stratum,title in [('small_le5pct','Małe ≤5%'),('gt50pct','Duże >50%')]:
        lines+=['',f'### {title}','']
        sub=metrics[(metrics.stratum==stratum)&(metrics.family=='all')]
        columns={'Wariant':'variant','FG IoU':'fg_iou','AP':'ap','Top-K':'topk_precision','Complete miss':'complete_miss','FP area':'false_positive_area'} if stratum=='small_le5pct' else {'Wariant':'variant','Clean IoU':'clean_iou','Internal BF1':'internal_boundary_f1','Hole recall':'hole_recall','Clean FPR':'clean_fpr'}
        lines+=table(sub,columns)
    folds=pd.read_csv(out/'per_fold.csv');lines+=['','### Wyniki per fold','']+table(folds,{'Rodzina':'family','Fold':'fold','Wariant':'variant','mIoU':'miou','AP':'ap'})
    no_e3=not summary['e3_gate']
    lines+=['','### Decyzja końcowa','',
            ('**NO-GO dla dalszej rozbudowy forensic fusion.** Proste warianty nie spełniły bramki awansu do E3; CNN nie trenowano. ' if no_e3 else '**E3 wymaga wykonania:** prosta fuzja przeszła bramkę. ')+
            'Bramka E3: delta mIoU nad dopasowaną kontrolą>0,005, dolna granica parowanego CI>0, co najmniej2 dodatnie foldy oraz small AP+0,005 lub large clean IoU+0,01. Przed pełnym wynikiem, po zaobserwowaniu niestabilności TrustMark, dopisano wymóg dodatniej delty względem oryginalnego baseline. Sama poprawa wobec słabej kontroli nie uzasadnia E3. Ten dodatkowy wymóg nie jest używany do wyboru wariantu ani progów.',
            '', 'Wybór DetectiveSAM po wcześniejszych diagnostykach i wielokrotne używanie tych samych outer-foldów oznaczają selekcję rozwojową. Bootstrap resampluje obrazy, nie treningi; przedstawiono seed42, nie niezależne potwierdzenie na3 seedach ani nowym zbiorze. Wynik nie uzasadnia wniosków o wszystkich możliwych fuzjach czy wszystkich modelach forensics.',
            '', '[Per-image fuzji](results/fusion/per_image_metrics.csv), [dobór ustawień](results/fusion/configurations.json), [sparowane delty](results/fusion/paired_deltas.csv), [galeria fuzji](results/fusion/gallery/index.html).']
    report=(HERE/'REPORT.md').read_text(encoding='utf-8')
    if '\n## Faza E:' in report:report=report.split('\n## Faza E:')[0]
    report=report.replace('Bramkę przechodzą: detectivesam. Wyniki nauczonej fuzji są raportowane osobno; samo przejście bramki nie dowodzi zysku.',
        'Bramkę przechodzi DetectiveSAM. Wykonano E1/E2 i dopasowane kontrole na ścisłych wewnętrznych cross-fitted mapach; wyniki i końcowa decyzja w sekcji Faza E poniżej.')
    report=report.replace('Możliwość skutecznego gate per obraz pozostaje nieudowodniona.',
        'W fazie E przetestowano również miękki gate per obraz; jego wynik należy oceniać względem watermark-only control, nie oracle.')
    report=report.replace('Bez pozytywnej bramki nie uczymy połączenia; nie twierdzimy, że przetestowano wszystkie możliwe fuzje.',
        'Dla DetectiveSAM wykonano E1/E2; wyniki poniżej pokazują, ile wnosi forensic ponad samą kalibrację. Nie testowano wszystkich możliwych fuzji.')
    if no_e3:report=report.replace('Wykonać minimalną, poprawnie cross-fitted próbę tylko modeli spełniających bramkę; decyzję końcową oprzeć na jej przewadze nad kontrolą.',
        'Próbę E1/E2 wykonano. Rekomendacja: zakończyć tę ścieżkę jako priorytet; nie potwierdzono wymaganej przewagi nad dopasowanymi kontrolami uzasadniającej E3.')
    executive=('**Wniosek: NO-GO dla dalszej rozbudowy obecnej fuzji.** Uruchomiono wszystkie3 oficjalne modele na510 obrazach (1530 map), a po diagnostykach wykonano E1/E2 DetectiveSAM z63 wewnętrznymi głowicami cross-fit. '
               'Baza OOF ma **0,50624 mIoU**, E1 **0,46816**, E2 **0,47946**. E2 wnosi nad własną kontrolę **+0,00352** (CI95 +0,00067…+0,00674), ale względem oryginalnej bazy **−0,02678** (CI95 −0,03559…−0,01855). '
               'Największy spadek pochodzi z niestabilnej kalibracji TrustMark, widocznej też w kontroli. Dla StegaStamp E2 ma **0,51629 vs0,51514**, delta+0,00114 z CI95 −0,00368…+0,00630. '
               'To słaby komplementarny sygnał w części obrazów, bez potwierdzonego przełomu w lokalizacji. Gate rozpoznaje kierunek w52 informatywnych przypadkach z dokładnością **51,9%**; balanced accuracy3 klas to **34,0%**. E3 nie trenowano.\n\n')
    if no_e3 and '**Wniosek: NO-GO' not in report:report=report.replace('\n\n','\n\n'+executive,1)
    (HERE/'REPORT.md').write_text(report+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    top=json.loads((OUT/'summary.json').read_text());top['status']='complete_no_go' if no_e3 else 'e3_required';top['fusion']='E1_E2_complete';top['fusion_summary']=summary
    save(OUT/'summary.json',top)
    gallery=out/'gallery';gallery.mkdir(exist_ok=True)
    base=df[df.variant=='baseline'].set_index('image_id');e1=df[df.variant=='E1_forensic'].set_index('image_id');e2=df[df.variant=='E2_forensic'].set_index('image_id')
    candidates=list((e1.miou-base.miou).sort_values().head(4).index)+list((e1.miou-base.miou).sort_values().tail(4).index)
    candidates+=list(e1[e1.gt_area<=.05].sort_values('ap',ascending=False).head(3).index)
    candidates+=list(e1[(e1.gt_area>.5)&(e1.hole_pixels>0)].sort_values('hole_recall',ascending=False).head(3).index)
    entries=[];cases=[]
    for n in dict.fromkeys(candidates):
        i=idx[n];r=records[i];family=r['family'];fold=r['fold'];gt=g[i]
        z=np.load(out/f'{family}_fold{fold}_E1_forensic.npz');j=z['names'].tolist().index(n);p=z['probability'][j];pred=p>=float(z['threshold'])
        zg=np.load(out/f'{family}_fold{fold}_E2_forensic.npz');pg=zg['probability'][zg['names'].tolist().index(n)]
        zc=np.load(out/f'{family}_fold{fold}_E1_control.npz');pc=zc['probability'][zc['names'].tolist().index(n)]
        f=np.load(OUT/'detectivesam'/f'{n}.npz')['probability'];rgb=np.asarray(Image.open(OUT/'inputs/rgb'/f'{n}.png'))
        overlay=np.asarray(Image.fromarray(rgb).resize((128,128))).astype(float)/255*.45
        overlay[pred&~gt]=[1,0,0];overlay[~pred&gt]=[0,.4,1]
        panels=[rgb,gt,b[i],f,1-f,pc,p,pg,overlay];labels=['RGB','GT','Baseline P','DetectiveSAM F','1-F','E1 control P','E1 forensic P','E2 gate P','E1 FP red / FN blue']
        fig,axes=plt.subplots(1,9,figsize=(24,3.2))
        for ax,im,title in zip(axes,panels,labels):
            ax.imshow(im,cmap='viridis' if title.endswith((' P',' F')) or title=='1-F' else 'gray',vmin=0 if im.ndim==2 else None,vmax=1 if im.ndim==2 else None);ax.set_title(title,fontsize=8);ax.axis('off')
        fig.suptitle(f'{n} {family} fold{fold} GT={gt.mean():.3f} | mIoU base {base.loc[n,"miou"]:.3f} E1 {e1.loc[n,"miou"]:.3f} E2 {e2.loc[n,"miou"]:.3f}')
        fig.tight_layout();fig.savefig(gallery/f'{n}.jpg',dpi=110);plt.close(fig)
        entries.append(f'<p>{n} {family}</p><img loading="lazy" src="{n}.jpg" style="width:100%">');cases.append(n)
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fusion OOF</title><h1>Fusion OOF</h1>'+''.join(entries),encoding='utf-8')
    save(gallery/'cases.json',cases)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,v in zip(axes,['E1_forensic','E2_forensic']):
        x=df[df.variant==v].set_index('image_id').loc[base.index]
        ax.scatter(base.miou,x.miou,s=10,alpha=.6);ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Baseline mIoU',ylabel=v+' mIoU')
    fig.tight_layout();fig.savefig(OUT/'fusion_scatter_examples.png',dpi=150);plt.close(fig)
    print('Fusion report complete, E3 gate=',summary['e3_gate'],flush=True)

if __name__=='__main__':main()
