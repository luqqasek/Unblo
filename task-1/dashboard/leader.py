"""Connect the best accepted public submission to its local evidence."""
import math


def best_submission(submissions):
    valid = [s for s in submissions if isinstance(s.get('score'), (int, float))
             and not isinstance(s['score'], bool) and math.isfinite(s['score'])]
    return max(valid, key=lambda s: (s['score'], s['id']), default=None)


def leader_evidence(submissions, experiments, dist):
    best = best_submission(submissions)
    if best is None:
        return None
    leader = dict(best, links=[], issues=[])
    experiment = best['experiment']
    entry = next((e for e in experiments if e['id'] == experiment), None)
    base = f'reports/{experiment}'

    def link(title, path):
        if (dist / path).is_file():
            leader['links'].append(dict(title=title, path=path))

    if entry:
        link('Research report and limitations', entry['report'])
    link('Submission results', f'{base}/SUBMISSION_REPORT.html')
    link('Validation predictions: successes and failures', f'{base}/results/gallery/index.html')
    link('Per-image results (CSV)', f'{base}/results/per_image_metrics.csv')
    link('Test prediction checks (JSON)', f'{base}/results/full_train/prediction_summary.json')
    link('Accepted submission receipt', best['receipt'])
    leader['detail'] = 'Highest accepted public mIoU across all three watermark families. Local validation scores below use separate protocols.'
    if experiment == 'synthetic_pretrain_2k' and entry:
        leader['note'] = 'StegaStamp: inverse-STN, 2000 synthetic training images, then real-image finetuning. PixelSeal and TrustMark predictions are retained from submission #1202.'
        leader['issues'] = [dict(detail=text) for text in (
            'Most of the local improvement comes from clean-background IoU; foreground IoU did not improve.',
            'Small watermarks remain difficult. Inspect the small-mask examples and per-image results.',
            'Finetuning has no confirmed local mIoU advantage over synthetic-only; synthetic-only was not submitted.',
            'Validation reuses 170 development images across three seeds. Gallery panels show OOF predictions, not the final test predictions.',
        )]
    elif experiment == 'next_ideas' and entry:
        leader['note'] = 'Native RGB StegaStamp adapted with real labels and synthetic replay. PixelSeal and TrustMark masks are retained from submission #2675. Training endpoints and postprocessing are selected on local validation.'
        leader['issues'] = [dict(detail=text) for text in (
            'Small watermarks and tiny clean holes remain difficult; average improvement does not eliminate complete misses.',
            'Local results use repeatedly inspected three-fold development validation and one seed, not an independent hidden-test estimate.',
            'The final all-170 refit has no held-out validation score. Inspect the fold results and failure gallery.',
        )]
    return leader
