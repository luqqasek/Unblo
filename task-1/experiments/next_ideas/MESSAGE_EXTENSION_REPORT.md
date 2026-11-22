# Bounded PS/TM continuation to450: measured results

Four workers on one4-A100 node, restored parent150 model/optimizer/RNG,300 additional updates, fixed original fold1, unchanged architecture/loss/LRs and effective batch16 (12 exact synthetic +4 real). Fixed128 replay versus existing online replay. No input augmentation or new embedding calls.

The TM primary comparison uses the unchanged integer90/100 verifier and caps accepted rectangles to15% image area at both endpoints. Historical uncapped maps are also retained for reproduction. It does not include the later empty-mask scan, so these TM numbers must not be called the full#2675 recipe. No test inference was run for these arms.

| Arm | mIoU150 | mIoU450 | Delta | Paired95% CI | Candidate |
|---|---:|---:|---:|---|---|
| ps-data-small | 0.570883 | 0.568250 | -0.002633 | [-0.007754181250164787, 0.0027355075323519445] | False |
| tm-data-small | 0.640957 | 0.631045 | -0.009911 | [-0.02260415785862517, 0.0012811463085438753] | False |
| ps-data-online | 0.565243 | 0.564761 | -0.000482 | [-0.006080947429322108, 0.004973951744691921] | False |
| tm-data-online | 0.646204 | 0.645036 | -0.001167 | [-0.009550353353515449, 0.007146990553430928] | False |

| Arm | Bucket/metric | n | Before150 | After450 | Misses before / after |
|---|---|---:|---:|---:|---|
| ps-data-small | fg_le1/fg_iou | 7 | 0.000000 | 0.000000 | 7/7 |
| ps-data-small | fg_1to5/fg_iou | 9 | 0.012724 | 0.000000 | 8/9 |
| ps-data-small | fg_5to20/fg_iou | 11 | 0.348457 | 0.343172 | 3/3 |
| ps-data-small | fg_gt20/fg_iou | 30 | 0.881765 | 0.886771 | 0/0 |
| ps-data-small | clean_le1/clean_iou | 4 | 0.000000 | 0.000000 | 4/4 |
| ps-data-small | clean_1to5/clean_iou | 10 | 0.075527 | 0.082018 | 6/5 |
| ps-data-small | wm_over50/clean_iou | 27 | 0.239856 | 0.232793 | 11/9 |
| tm-data-small | fg_le1/fg_iou | 3 | 0.000000 | 0.000000 | 3/3 |
| tm-data-small | fg_1to5/fg_iou | 18 | 0.516439 | 0.516360 | 4/3 |
| tm-data-small | fg_5to20/fg_iou | 6 | 0.339047 | 0.347280 | 1/0 |
| tm-data-small | fg_gt20/fg_iou | 30 | 0.864928 | 0.859538 | 0/0 |
| tm-data-small | clean_le1/clean_iou | 4 | 0.000000 | 0.000000 | 4/4 |
| tm-data-small | clean_1to5/clean_iou | 5 | 0.111641 | 0.040573 | 2/2 |
| tm-data-small | wm_over50/clean_iou | 26 | 0.247981 | 0.208580 | 6/7 |
| ps-data-online | fg_le1/fg_iou | 7 | 0.000000 | 0.000000 | 7/7 |
| ps-data-online | fg_1to5/fg_iou | 9 | 0.006107 | 0.000000 | 8/9 |
| ps-data-online | fg_5to20/fg_iou | 11 | 0.344991 | 0.307592 | 3/4 |
| ps-data-online | fg_gt20/fg_iou | 30 | 0.887472 | 0.885145 | 0/0 |
| ps-data-online | clean_le1/clean_iou | 4 | 0.000000 | 0.000000 | 4/4 |
| ps-data-online | clean_1to5/clean_iou | 10 | 0.061511 | 0.067873 | 7/7 |
| ps-data-online | wm_over50/clean_iou | 27 | 0.215163 | 0.232021 | 12/11 |
| tm-data-online | fg_le1/fg_iou | 3 | 0.000000 | 0.000000 | 3/3 |
| tm-data-online | fg_1to5/fg_iou | 18 | 0.535927 | 0.531834 | 5/4 |
| tm-data-online | fg_5to20/fg_iou | 6 | 0.274145 | 0.269547 | 2/0 |
| tm-data-online | fg_gt20/fg_iou | 30 | 0.869723 | 0.875642 | 0/0 |
| tm-data-online | clean_le1/clean_iou | 4 | 0.000000 | 0.000000 | 4/4 |
| tm-data-online | clean_1to5/clean_iou | 5 | 0.135631 | 0.129578 | 1/1 |
| tm-data-online | wm_over50/clean_iou | 26 | 0.259375 | 0.258154 | 5/6 |

Arms passing the predeclared +0.005 gain, positive paired lower confidence bound and minority guards: none. The confidence intervals do not correct for repeated fold reuse or four-arm selection. The best accepted submission remains#3054/public0.6677 unless a later actual server receipt establishes otherwise.

Orchestrator wall seconds: 304.06. Additional optimizer updates1200, total19200 image views (14400 synthetic replay +4800 real draws). Parent probability reproduction is checked within1e-4 before training; logs include measured throughput and allocated memory. No corpus expansion was performed.
