"""Summarize source/geometry diagnostics without fitting generator distributions."""
import json
from pathlib import Path
import numpy as np
from experiments.native_localizer.common import atomic_json
P=Path(__file__).resolve().parent/'results'

def stats(rows,keys):
 out={'n':len(rows)}
 for k in keys:
  v=[r[k] for r in rows if r.get(k) is not None]
  out[k]={'n':len(v),'mean':float(np.mean(v)) if v else None,'p10_p50_p90':np.quantile(v,[.1,.5,.9]).tolist() if v else None}
 return out

def main():
 result={}
 for f in ['pixelseal','trustmark']:
  root=P/'runs'/f
  if not (root/'synthetic_geometry_signal.json').exists():continue
  rows=json.loads((root/'synthetic_geometry_signal.json').read_text());real=json.loads((root/'real_geometry.json').read_text());d={}
  keys=['bbox_fill','components','width_p50_grid','boundary_to_area','texture_gradient_rgb_units']
  for name,rr in [('real',real)]+[(v,[x for x in rows if x['variant']==v]) for v in ['control','objects']]:
   d[name]=stats([x for x in rr if 0<x['fg_fraction']<1],keys)
  for variant in ['control','objects']:
   for name,lo,hi in [('fg_le1',0,.01),('fg_1to5',.01,.05),('fg_gt5',.05,1)]:
    rr=[r for r in rows if r['variant']==variant and lo<r['fg_fraction']<=hi]
    d[variant+'_'+name]=stats(rr,['global_bit_accuracy','rms','zero_pixel_fraction','residual_rms_fg_cells','zero_pixel_fraction_fg_cells'])
   for regime in sorted({r['regime'] for r in rows}):
    d[variant+'_'+regime]=stats([r for r in rows if r['variant']==variant and r['regime']==regime],['global_bit_accuracy','rms','zero_pixel_fraction'])
  enc=json.loads((root/'encoding_signal.json').read_text())
  d['encoding_regimes']={regime:stats([r['residual_uint8'] for r in enc if r['regime']==regime],['rms','zero_fraction','zero_pixel_fraction']) for regime in sorted({r['regime'] for r in enc})}
  result[f]=d
 atomic_json(P/'audit_summary.json',result)
if __name__=='__main__':main()
