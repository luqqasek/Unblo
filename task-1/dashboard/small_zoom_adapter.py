import json,shutil
def add_rows(experiments,dist,row):
    from polish import COPY
    COPY.update({'small zoom':'Małe watermarki: crop-and-decode','Existing 3-fold OOF + frozen GT-free crop correction; development':'Istniejące 3-fold OOF + zamrożona korekta crop bez GT; rozwój','Synthetic-only / 170 real images':'Tylko syntetyki / 170 realnych obrazów','Fixed 3-fold adapted OOF / 170 real images':'Stała adaptacja 3-fold OOF / 170 realnych obrazów','Single fixed fold1 screening / 57 real images':'Screening jednego ustalonego foldu1 / 57 realnych obrazów','Tiny exact rectangles':'Dokładne małe prostokąty','Tiny rectangles + Y residual auxiliary':'Małe prostokąty + pomocniczy residual Y','Matched continued baseline':'Dopasowana kontynuacja baseline'})
    root=experiments/'small_zoom';path=root/'results/measurements.json'
    if not path.exists():return
    for m in json.loads(path.read_text(encoding='utf-8')):
        g=m['groups'];variant={'tiny':'Tiny exact rectangles','tiny_aux':'Tiny rectangles + Y residual auxiliary','control':'Matched continued baseline'}.get(m['variant'],m['variant'])
        row('small_zoom',m['family'],variant,m['protocol'],g['all']['miou'],f"FG IoU <=1%: {g['le1']['fg_iou']:.5f}; 1-5%: {g['1to5']['fg_iou']:.5f}; 5-20%: {g['5to20']['fg_iou']:.5f}; >20%: {g['gt20']['fg_iou']:.5f}; clean holes IoU: {g['clean_holes']['clean_iou']:.5f}. Development validation; public score is separate.")
    dest=dist/'reports/small_zoom/results';dest.mkdir(parents=True,exist_ok=True)
    for name in ['measurements.json','latest_oof_diagnosis.json','runs_summary.json','multiscale_summary.json','wide_summary.json','finalscale_summary.json']:
        if (root/'results'/name).exists():shutil.copyfile(root/'results'/name,dest/name)
    submission=root/'results/submission'
    if submission.exists():
        target=dest/'submission';target.mkdir(parents=True,exist_ok=True)
        for path in submission.glob('*.json'):shutil.copyfile(path,target/path.name)
