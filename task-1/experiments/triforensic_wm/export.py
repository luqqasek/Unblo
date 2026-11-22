"""Export a completed, gate-passing EMA as a portable inference checkpoint."""
import argparse,json
from pathlib import Path
import torch
from experiments.native_localizer.common import atomic_json,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();cfg=json.loads((a.run/'config.json').read_text());done=json.loads((a.run/'completed.json').read_text())
    target=cfg.get('target_steps',cfg['steps'])
    assert done['stop'] in ('steps_complete','user_authorized_early_stop'),'Incomplete run is not a frozen model'
    assert done['step']==target or done['stop']=='user_authorized_early_stop'
    assert done['gate'] and done['gate']['passed'],'Do not export a failed gate as a selected system'
    state=torch.load(a.run/'resume.pt',map_location='cpu',weights_only=False);dest=a.run/'ema-model.pt';assert not dest.exists(),'Preserve prior export'
    assert state['step']==done['step']
    production=bool(cfg.get('all_real',False))
    if production:assert cfg['real_training_images']==170 and cfg['real_validation_images']==0
    torch.save({'model':state['ema'],'family':cfg['family'],'step':state['step'],'config':cfg},dest)
    atomic_json(a.run/'frozen_inference.json',{'family':cfg['family'],'checkpoint':str(dest),'checkpoint_sha256':sha256(dest),'threshold':.5,'tta':False,'native_rgb':512,'mask_size':128,'precision':'FP32 image/filter/first projection;BF16 deeper layers','trustmark_verifier_correct_bits':90,'max_verified_area':.15,'veto_threshold':2.,'selection':'Endpoint selected on reused fixed-fold validation; no threshold tuning','selected_updates':done['step'],'schedule_total_updates':cfg['steps'],'all_real_refit':production,'training_scope':'All170 real images with synthetic replay' if production else '113 real fold1 images; screening only, not a final all170 refit','screening_source':cfg.get('screening_source'),'test_inference_allowed':production})
    print(dest)
if __name__=='__main__':main()
