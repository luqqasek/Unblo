"""No-training PCMV checks: existing GT-view evidence and exact scoring/null arithmetic."""
import json,math,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results';OUT.mkdir(exist_ok=True)
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def score(p,y):
 p=np.asarray(p);y=np.asarray(y);return float(np.mean([np.mean((p==c)&(y==c))/np.mean((p==c)|(y==c)) if np.any((p==c)|(y==c)) else 1 for c in (0,1)]))
y=np.array([1,0,0,0,1,0,0,0,1,1,0,1,0,1,0,0,0,1,1,1]);base=np.array([1,1,0,0,0,0,1,1,1,0,0,1,0,0,1,1,1,0,0,1]);A=[0,6,7,14,2,9,11,10];B=[16,19,15,18,1,4,8]
a=base.copy();a[A]=0;b=base.copy();b[B]=0;ab=a.copy();ab[B]=0
scores={k:score(p,y) for k,p in [('base',base),('A',a),('B',b),('A_then_B',ab)]}
assert not set(A)&set(B) and scores['A']>scores['base'] and scores['B']>scores['base'] and scores['A_then_B']<scores['base']
save('toggle_nonadditivity.json',{'scope':'algebraic counterexample, not a dataset result','truth':y.tolist(),'base':base.tolist(),'carve_A':A,'carve_B':B,'scores':scores,'independent_predicted_gain_sum':scores['A']+scores['B']-2*scores['base'],'actual_joint_gain':scores['A_then_B']-scores['base']})
null=[]
for bits,correct in [(256,148),(256,149),(256,174),(100,58),(100,68)]:
 p=sum(math.comb(bits,k) for k in range(correct,bits+1))/2**bits
 null.append({'bits':bits,'correct':correct,'accuracy':correct/bits,'binary_alignment_z':(2*correct-bits)/math.sqrt(bits),'exact_fair_coin_one_sided_p':p,'fwer_80_independent_probes':-math.expm1(80*math.log1p(-p)),'fwer_320_independent_probes':-math.expm1(320*math.log1p(-p))})
save('null_arithmetic.json',{'rows':null,'limits':'Independent-probe FWER is an illustration, not a calibration for correlated/adaptive proposals. Fixed-message permutations are a different null from independent random signs. Real search needs full-pipeline empirical null.','fixed_balanced_message_constant_v_example':{'bits':256,'v':'all ones','signed_message':'128 plus ones,128 minus ones','permutation_statistic_variance':0,'independent_sign_statistic_variance':1}})
allresults={};sources={}
for family in ('pixelseal','trustmark','stegastamp'):
 p=ROOT/f'experiments/next_ideas/results/oracle/{family}.json';d=json.loads(p.read_text());sources[family]=hashlib.sha256(p.read_bytes()).hexdigest();rows=d['per_image'];summ={}
 for group,condition in [('fg_le1',lambda r:0<r['true_area']<=.01),('fg_1to5',lambda r:.01<r['true_area']<=.05),('fg_le5',lambda r:0<r['true_area']<=.05)]:
  rr=[r for r in rows if condition(r)];views={}
  for view in ('full','GT_bbox_direct_native','outside_GT_blurred','outside_GT_mean','GT_blurred'):
   vals=[r['views'][view] for r in rr];z=[(v['soft_agreement']-v['wrong_message_mean'])/max(v['wrong_message_std'],1e-12) for v in vals]
   views[view]={'n':len(vals),'mean_bit_accuracy':float(np.mean([v['bit_accuracy'] for v in vals])),'median_empirical_standardized_alignment':float(np.median(z)),'n_z_ge2_5':int(sum(v>=2.5 for v in z)),'n_z_ge3_5':int(sum(v>=3.5 for v in z)),'n_empirical_p_le_0_05':sum(v['empirical_wrong_p']<=.05 for v in vals),'n_bit_accuracy_ge90':sum(v['bit_accuracy']>=.9 for v in vals)}
  summ[group]=views
 allresults[family]=summ
save('cached_oracle_reanalysis.json',{'families':allresults,'source_sha256':sources,'scope':'GT-defined support diagnostics using archived soft tanh(logits/2) alignment against169 other real messages; not deployable proposals, not PCMV exact bias-corrected z, not proposal AUC.','null_resolution':1/170,'limits':'No matched within-image negative proposals, no clean-bias fitting, no JND-stratified statistic, no search-level calibration. Existing mean fill uses image mean, not boundary mean. These results cannot pass the PCMV Stage0 gate.'})
print(json.dumps({'nonadditivity':scores,'null_148_of_256':null[0],'small_fg':{f:{v:r['fg_le5'][v] for v in ('GT_bbox_direct_native','outside_GT_mean')} for f,r in allresults.items()}},indent=2))
