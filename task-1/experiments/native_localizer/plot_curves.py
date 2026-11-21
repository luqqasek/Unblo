"""Plot actual saved updates; the real score is one final point, never a curve."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import jsonl


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    train=jsonl(a.run/"train.jsonl"); dev=jsonl(a.run/"dev.jsonl")
    summary=json.loads(a.summary.read_text()); batch=summary["effective_batch"]
    fig,axes=plt.subplots(2,1,figsize=(9,7),layout="constrained",sharex=True)
    axes[0].plot([r["seen_views"] for r in train],[r["train_batch_miou"] for r in train],
                 color="0.65",label="Train batch (changing examples)")
    axes[0].plot([r["update"]*batch for r in dev],[r["miou"] for r in dev],"o-",
                 color="#1468b3",label="Fixed unseen-source/message synthetic dev")
    axes[0].scatter([summary["training_views"]],[summary["real_validation_miou"]],
                    color="#c33826",marker="D",s=65,label="Real 170: final checkpoint only",zorder=4)
    axes[0].set(ylim=(0,1),ylabel="Official mean per-image two-class mIoU",
                title="StegaStamp native RGB pilot: measured learning curves")
    axes[0].legend(loc="lower right",fontsize=8); axes[0].grid(alpha=.2)
    for key,label in (("seg","Segmentation"),("local","Local auxiliary (unweighted)"),("pair","Pair auxiliary (unweighted)")):
        axes[1].plot([r["seen_views"] for r in train],[r[key] for r in train],label=label)
    axes[1].set(xlabel="Actual training image views (batch64)",ylabel="Loss")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    fig.savefig(a.out,dpi=160); plt.close(fig)


if __name__=="__main__":
    main()
