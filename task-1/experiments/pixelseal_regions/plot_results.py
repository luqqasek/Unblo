"""Render the measured loss and recognition curves for the experiment journal."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out = Path(__file__).parent / 'results'
    train = [json.loads(line) for line in (out / 'train.jsonl').read_text().splitlines()]
    gates = [json.loads(p.read_text()) for p in sorted(out.glob('gate-*.json'))]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
    ax[0].plot([r['step'] for r in train], [r['loss'] for r in train], color='#27638d')
    ax[0].set(title='Training loss falls', xlabel='Optimizer update', ylabel='Mean loss / last 25 updates')
    for group, label in [('all', 'All held-out pairs'), ('real_clean', 'Real clean-source pairs'), ('width_le24', 'Shorter side ≤24 px')]:
        ax[1].plot([r['step'] for r in gates], [r['recognition'][group]['auc'] for r in gates], marker='o', label=label)
    ax[1].axhline(.5, color='gray', linestyle='--', linewidth=1)
    ax[1].set(title='Held-out recognition stays near chance', xlabel='Optimizer update', ylabel='Known-region ROC-AUC', ylim=(.45, .60))
    ax[1].legend(fontsize=8)
    fig.suptitle('PixelSeal native region screen — no validation mIoU gain', fontsize=11)
    fig.savefig(out / 'learning_curves.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
