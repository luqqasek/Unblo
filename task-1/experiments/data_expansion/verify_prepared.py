"""Supplementary cross-dataset pHash screen before any exact embedding."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    old=[]
    for split in ('train','dev','locked'):
        for line in (a.root/f'sources/coco2017/{split}.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r.get('phash'):old.append((r['source_id'],split,int(r['phash'],16)))
    folder=a.root/'data-expansion/pipe-prepared-v2';hits=[];count=0
    for split in ('train','dev','locked'):
        for line in (folder/f'{split}.jsonl').read_text().splitlines():
            r=json.loads(line);h=int(r['phash'],16);count+=1
            for n,other,ph in old:
                if (h^ph).bit_count()<=4:
                    hits.append({'new':r['source_id'],'new_split':split,'existing':n,'existing_split':other})
    result={'existing_photos_screened':len(old),'candidate_photos':count,'near_duplicate_candidates':hits,
            'method':'pHash Hamming<=4; supplementary, not exhaustive','passed':not hits}
    (folder/'old_source_duplicate_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));assert not hits


if __name__=='__main__':main()
