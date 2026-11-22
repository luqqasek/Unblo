"""Plot measured validation endpoints; no interpolation is evaluated as evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent/'results'
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,prefix,title in zip(axes,['ps','tm'],['PixelSeal','TrustMark']):
    for arm,label in [('small','128 fixed compositions'),('online','Larger pool, online compositions')]:
        folder=root/f'{prefix}-data-{arm}';x=[];y=[]
        for step in [0,50,150]:
            p=folder/f'correction-{step}.json'
            if p.exists():x.append(step);y.append(json.loads(p.read_text())['summary']['all']['miou'])
        ax.plot(x,y,'o-',label=label)
    ax.set_title(title+' — fixed validation fold, n=57');ax.set_xlabel('Additional optimizer updates');ax.set_ylabel('Official per-image mIoU');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Full-model fine-tuning: 75% synthetic replay + 25% real training images')
fig.savefig(root/'data_example_validation_curves.png',dpi=160);plt.close(fig)
