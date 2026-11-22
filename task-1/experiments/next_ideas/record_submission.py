"""Record an actual server receipt; never submit or infer a score."""
import argparse,json
from pathlib import Path
from experiments.native_localizer.common import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--extended',action='store_true');a=p.parse_args()
    here=Path(__file__).parent;result=here/('results/extended' if a.extended else 'results');out=result/'submission'
    receipt=json.loads((out/'server_receipt.json').read_text());response=receipt['response'];assert receipt['http_status']==200 and response['status']=='success'
    check=json.loads((out/'assembly_validation.json').read_text());assert check['sha256']==receipt['artifact_sha256']
    cfg=json.loads((result/'selected_recipe.json').read_text());assert check['changed_images']['pixelseal']==check['changed_images']['trustmark']==0
    normalized=dict(receipt,response=json.dumps(response));atomic_json(out/'upload_receipt.json',normalized)
    title='450-step continuation' if a.extended else '150-step real adaptation'
    report=f'''# StegaStamp {title}: submission #{response['submission_id']}

Accepted public mIoU: **{response['score']:.4f}**. Previous best reported by the server: **{response['previous_best']:.4f}**. Server improvement flag: **{response['improved']}**.

The artifact changes only StegaStamp. All PixelSeal and TrustMark masks are identical to submission #2675. Hidden final-set performance and per-family public scores are unavailable.

## Validation and selection

StegaStamp OOF mIoU: **{cfg['validation_miou_final']:.6f}**, including the fixed empty/full-mask fallback. This is three-fold development validation on 170 real images, one seed, with repeatedly inspected folds. The all-170 refit has no held-out validation score. Public scores were not used to choose the training endpoint, threshold or fallback.

Raw FP32 native 512-pixel model, threshold 0.5, no TTA. All backbone, local stem and decoder weights are trainable. AdamW uses backbone LR 3e-6 and new-module LR 1e-5; every effective batch contains 12 exact synthetic and 4 real training views. The original 2,500-source synthetic pool is reused without new embedding calls or post-quantization augmentation.

The fixed fallback changes only completely empty/full predictions, selecting the 164 highest minority-probability cells. It does not use GT at inference and does not establish that all hidden images contain both classes.

## Artifact checks

- 1,020 unique official test names, binary uint8 masks of shape [1020, 128, 128].
- File size: {check['bytes']:,} bytes, below 20 MB.
- SHA-256: `{check['sha256']}`.
- Changed images relative to #2675: `{json.dumps(check['changed_images'])}`.
- Receipt timestamp: {receipt['timestamp']}.

Exact configuration, checkpoint hash and shard hashes are in `assembly_validation.json` and the frozen `selected_recipe.json`. Credentials were read from stdin and are absent from these files. Prior models and submissions are preserved.
'''
    dest=here/('SUBMISSION_450_REPORT.md' if a.extended else 'SUBMISSION_REPORT.md');dest.write_text(report)
    print(json.dumps({'submission_id':response['submission_id'],'public_miou':response['score'],'improved':response['improved'],'report':str(dest)}))


if __name__=='__main__':main()
