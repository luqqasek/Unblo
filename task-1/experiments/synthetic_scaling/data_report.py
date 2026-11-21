"""Inspect retained examples and compare area distributions, without training."""
import zipfile
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scaling_common import *

def main():
    source=module('scaling_data_report_source',OLD/'prepare.py')
    with zipfile.ZipFile(OLD/'results/sources/annotations_trainval2017.zip') as z:
        coco=json.loads(z.read('annotations/instances_val2017.json'))
    annotations={}
    for a in coco['annotations']:annotations.setdefault(a['image_id'],[]).append(a)
    fig,axes=plt.subplots(3,4,figsize=(12,9))
    with zipfile.ZipFile(OLD/'results/sources/val2017.zip') as archive:
        for i,m in enumerate(METHODS):
            rows=json.loads((OUT/'data'/m/'manifest.json').read_text());r=rows[3]
            filename=r.get('source_filename') or next(a['file_name'] for a in coco['images'] if a['id']==r['coco_id'])
            cover,region,_,_=source.cover_and_mask(archive.read('val2017/'+filename),annotations.get(r['coco_id'],[]),np.random.default_rng(20261124),'rectangle')
            path=(OLD/'results/images'/f'{r["image_id"]}.png') if m=='stegastamp' else OUT/'data'/m/'examples/0003_image.png'
            image=np.asarray(Image.open(path).convert('RGB'))
            mask=np.load(OUT/'data'/m/'targets.npy',mmap_mode='r')[3]
            delta=np.mean(abs(image.astype(float)-cover.astype(float)),axis=2)
            for ax,data,title in zip(axes[i],[cover,image,mask,delta],[m+' cover','Synthetic image','GT 128x128','Mean absolute RGB delta']):
                ax.imshow(data,cmap='magma' if data.ndim==2 else None);ax.set_title(title);ax.axis('off')
    fig.suptitle('Same COCO cover and patch mask; three actual watermark encoders')
    fig.tight_layout();fig.savefig(OUT/'generation_examples.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.5));stats=[]
    for ax,m in zip(axes,METHODS):
        rows=json.loads((OUT/'data'/m/'manifest.json').read_text())
        real=np.load(HERE.parent/f'frozen_extractor/results/cv3/{m}_oof.npz')['target'].mean(axis=(1,2))
        for name,areas in [('Real',real),('Synthetic 2k',np.array([rows[i]['area'] for i in train_ids(2000)])),('Synthetic 4k',np.array([rows[i]['area'] for i in train_ids(4000)]))]:
            ax.hist(areas,bins=np.linspace(0,1,21),weights=np.ones(len(areas))/len(areas),histtype='step',linewidth=1.5,label=name)
            stats.append(dict(method=m,pool=name,n=len(areas),mean_area=float(areas.mean()),small_le5pct=float((areas<=.05).mean()),empty=float((areas==0).mean()),full=float((areas==1).mean())))
        ax.set_title(m);ax.set_xlabel('GT watermark fraction');ax.set_ylabel('Image fraction');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'generation_areas.png',dpi=140);plt.close(fig)
    save(OUT/'generation_area_summary.json',stats)
    print('Saved generation panels and area comparisons',flush=True)

if __name__=='__main__':main()
