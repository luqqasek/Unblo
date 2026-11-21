"""Measured native pilots and selected hybrid; never expose tensors in dashboard."""
import json
import shutil

def add_rows(experiments,dist,row):
    from polish import COPY
    COPY.update({'Native RGB pilot, 200 updates':'Pilot z RGB, 200 aktualizacji',
                 'Synthetic-only transfer, 170 real images':'Transfer bez realnego treningu, 170 obrazów',
                 'Selected three-seed ensemble':'Wybrany ensemble trzech seedów',
                 'Saved development OOF, 170 images':'Zapisane rozwojowe OOF, 170 obrazów',
                 'New StegaStamp, frozen threshold 0.5':'Nowy StegaStamp, stały próg 0,5',
                 'Native RGB and best legacy models':'Model z RGB i najlepsze wcześniejsze modele',
                 'native hybrid submission':'Najlepsze modele osobno dla metod'})
    base=experiments/'native_hybrid_submission/results'
    path=base/'validation_comparison.json'
    if not path.exists():return
    data=json.loads(path.read_text(encoding='utf-8'))
    for method,value in data['native_pilot'].items():
        row('native_hybrid_submission',method,'Native RGB pilot, 200 updates','Synthetic-only transfer, 170 real images',value)
    for method,value in data['selected_validation'].items():
        row('native_hybrid_submission',method,'New StegaStamp, frozen threshold 0.5' if method=='stegastamp' else 'Selected three-seed ensemble',
            'Synthetic-only transfer, 170 real images' if method=='stegastamp' else 'Saved development OOF, 170 images',value,
            'Development selection; different family-specific protocols. Not an unbiased test estimate.')
    target=dist/'reports/native_hybrid_submission/results'; target.mkdir(parents=True,exist_ok=True)
    for name in ('validation_comparison.json','assembly_validation.json','frozen_recipe.json','prediction_summary.json','server_receipt.json'):
        if (base/name).exists():shutil.copyfile(base/name,target/name)
