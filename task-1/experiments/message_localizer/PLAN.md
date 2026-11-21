# Message-aware improvement experiment

Before results: original exact PS/TM decoder initialization; native512 features
fused with native-decoder-size features (PS256/TM245); dot/cosine/soft-accuracy
maps from known bits; learned local RGB/Y/HPF context and original PS JND
function applied to the observed image. Progressive128-channel decoder.
No gradient/reembedding feature input: the diagnostic showed near-random maps.

Synthetic pilot: 400updates,batch32,50frozen-extractor warmup updates; then
joint learning at encoder2e-5/new3e-4 with warmup and cosine floor20%.
AllFP32; allBN running stats frozen; original embedders unchanged.
Existing2500 exact encoded train sources +128dev sources; no competition
labels in this pilot. Irregular procedural masks and holes enabled immediately.
No COCO instance transfers or inpainting in this iteration.
Per-image BCE+both-classDice, pair ranking ramped to0.1; no dense bit loss.
Evaluate fixed final raw weights at0.5 on all170 real images per family.

Follow-up fixed before synthetic-only real results: three original folds,
150updates/fold,batch32=24synthetic+8real outer-train views, encoder2e-6/new2e-5,
threshold0.5; no checkpoint/threshold tuning on outer validation.
Use original reconstructed folds; source-sibling limitations remain.

Also predeclare a fixed equal-probability ensemble with incumbent PS/TM
OOF ensembles. Compare 0.5*(p_new+p_old) >= 0.5*(0.5+t_old), without searching
weights or thresholds on these OOF labels. Repeated development selection
still limits claims; no new leaderboard submission is part of this experiment.
