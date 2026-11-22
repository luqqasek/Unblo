"""Keep synthetic-only transfer and adapted OOF as separate score protocols."""
import json,shutil

def add_rows(experiments,dist,row):
 from polish import COPY
 COPY.update({'region data':'Dane obiektowe i małe pozostałości','Object recipe, matched extra100':'Dane obiektowe, dodatkowe100 kroków','Equal-area position pairs, extra100':'Pary pozycyjne o równym polu, dodatkowe100 kroków','Original generator, matched continuation':'Obecny generator, dopasowana kontynuacja','Object masks and remnants':'Maski obiektów i pozostałości','Synthetic-only transfer / 170 real images':'Transfer syntetyczny / 170 realnych obrazów','Matched 3-fold OOF / seed42 / 170 images':'Dopasowane 3-fold OOF / seed42 / 170 obrazów'})
 base=experiments/'region_data/results';path=base/'summary.json'
 if not path.exists():return
 d=json.loads(path.read_text(encoding='utf-8'))
 for family,variants in d.items():
  for stage in ['synthetic','adapt','position']:
   for variant in (['objects_plus100','position_pairs100'] if stage=='position' else ['control','objects']):
    key=stage+'_'+variant
    if key not in variants:continue
    s=variants[key];row('region_data',family,{'control':'Original generator, matched continuation','objects':'Object masks and remnants','objects_plus100':'Object recipe, matched extra100','position_pairs100':'Equal-area position pairs, extra100'}[variant],'Synthetic-only transfer / 170 real images' if stage!='adapt' else 'Matched 3-fold OOF / seed42 / 170 images',s['all']['miou'],f"FG IoU <=1%: {s['fg_le1']['fg_iou']:.5f}; 1-5%: {s['fg_1to5']['fg_iou']:.5f}. Fixed threshold0.5; no new submission.")
 target=dist/'reports/region_data/results';target.mkdir(parents=True,exist_ok=True)
 for name in ['summary.json','audit_summary.json']:
  if (base/name).exists():shutil.copyfile(base/name,target/name)
