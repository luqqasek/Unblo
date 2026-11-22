"""Pinned official Noiseprint++ source/weights; keep original license notices."""
import hashlib,json,urllib.request,urllib.parse
from pathlib import Path
COMMIT='ae54475df6f41a491d7615100feb19263dec13f7'
def main():
    out=Path(__file__).resolve().parent/'vendor';out.mkdir(exist_ok=True);records=[]
    files={'DnCNN.py':'TruFor_train_test/lib/models/DnCNN.py','noiseprint.th':'TruFor_train_test/pretrained_models/noiseprint++/noiseprint++.th','LICENSE.txt':'TruFor_train_test/LICENSE.txt'}
    for name,path in files.items():
        url=f'https://raw.githubusercontent.com/grip-unina/TruFor/{COMMIT}/'+urllib.parse.quote(path,safe='/');data=urllib.request.urlopen(url).read();assert not data.startswith(b'version https://git-lfs');(out/name).write_bytes(data);records.append({'file':name,'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    (out.parent/'noiseprint_provenance.json').write_text(json.dumps({'commit':COMMIT,'authors':'Guillaro, Cozzolino, Sud, Dufour, Verdoliva; TruFor CVPR2023, GRIP-UNINA','files':records},indent=2)+'\n');print('Verified downloaded files',[(r['file'],r['bytes']) for r in records])
if __name__=='__main__':main()
