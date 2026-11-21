"""Explicit one-time cleanup; inspect manifest first, then pass --apply. No wildcard trees."""
import argparse,hashlib,json,os,subprocess,zipfile,gzip,shutil,time,zlib
from pathlib import Path
ROOT=Path('/p/project1/training2666/pszenny1/native-localizer')

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
 assert not subprocess.check_output(['squeue','-h','-u','pszenny1'],text=True).strip(),'Active jobs: do not clean'
 out=ROOT/'cleanup-20261122';out.mkdir(exist_ok=True)
 if (out/'completed.json').exists():raise RuntimeError('Already completed; inspect receipt')
 src=ROOT/'sources/coco2017/bulk';archive=src/'train2017.zip'
 checks={}
 with zipfile.ZipFile(archive) as z:
  entries=[x for x in z.infolist() if not x.is_dir()]
  expected={Path(x.filename).name for x in entries}
  actual=set(os.listdir(src/'train2017'))
  assert expected<=actual and len(entries)==118287,(len(expected),len(actual))
  checks['coco_extracted_name_coverage']=len(entries)
  # Exact CRC of deterministic spaced samples, not a claim of exhaustive pixel verification.
  samples=[entries[i] for i in [0,len(entries)//4,len(entries)//2,3*len(entries)//4,len(entries)-1]]
  for e in samples:
   data=(src/e.filename).read_bytes();assert len(data)==e.file_size and zlib.crc32(data)==e.CRC
  checks['coco_sample_crc_verified']=[e.filename for e in samples]
 protected=[ROOT/'runs/stegastamp/pilot-30min/last.pt']
 for family in ['pixelseal','trustmark']:
  protected += [ROOT/f'message-runs/{family}/{run}/last.pt' for run in ['matched-v1','matched-v1-full-refit']]
  protected += [ROOT/f'message-runs/{family}/matched-v1-adapt/fold{f}/last.pt' for f in [1,2,3]]
 retained=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in protected]
 assert retained[0]['sha256']=='2b633631361e2bfee583a438dc20db9da8dc7701895ccdc4b0a531012aedcdd5'
 candidates=[(archive,'Redundant download archive; extracted image names complete and sample CRC verified')]
 chunks=src/'train2017.chunks'
 for f in sorted(chunks.iterdir()):
  assert f.is_file() and not f.is_symlink();candidates.append((f,'Download chunks superseded by merged/extracted COCO'))
 for family in ['pixelseal','trustmark']:
  for name in ['last.pt','best.pt']:candidates.append((ROOT/f'runs/{family}/pilot-30min/{name}','Rejected native RGB pilot; metrics and configs retained'))
 candidates += [(ROOT/'runs/stegastamp/overfit-fp32/last.pt','Debug memorization-only weights; metrics retained'),(ROOT/'runs/stegastamp/pilot-30min/best.pt','Non-selected synthetic-dev checkpoint; deployed raw last checkpoint preserved')]
 records=[]
 for path,reason in candidates:
  assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT)
  st=path.stat();assert st.st_nlink==1,('Hardlink accounting needs separate review',path)
  records.append({'path':str(path.relative_to(ROOT)),'bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'reason':reason,'sha256':sha(path) if path.suffix=='.pt' else None})
 plan={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'checks':checks,'retained_checkpoints':retained,'delete':records,'logical_bytes_to_remove':sum(r['bytes'] for r in records),'allocated_bytes_to_remove':sum(r['allocated_bytes'] for r in records)}
 (out/'plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps({k:plan[k] for k in ['logical_bytes_to_remove','allocated_bytes_to_remove']}) ,flush=True)
 if not a.apply:return
 for r in records:
  path=ROOT/r['path'];assert path.stat().st_size==r['bytes'];path.unlink()
 compressed=[]
 for path in sorted((ROOT/'logs').iterdir()):
  if path.suffix not in ['.out','.err'] or not path.is_file():continue
  target=Path(str(path)+'.gz')
  if target.exists():continue
  size=path.stat().st_size
  with path.open('rb') as source,gzip.open(target,'wb') as dest:shutil.copyfileobj(source,dest)
  with gzip.open(target,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==sha(path)
  path.unlink();compressed.append({'path':str(path.relative_to(ROOT)),'before':size,'after':target.stat().st_size})
 for r in retained:assert (ROOT/r['path']).exists() and (ROOT/r['path']).stat().st_size==r['bytes']
 for r in records:assert not (ROOT/r['path']).exists()
 plan.update(completed=True,compressed_logs=compressed,log_logical_bytes_recovered=sum(x['before']-x['after'] for x in compressed))
 (out/'completed.json').write_text(json.dumps(plan,indent=2));print('CLEANUP_COMPLETED',flush=True)
if __name__=='__main__':main()
