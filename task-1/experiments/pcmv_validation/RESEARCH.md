# PCMV: source and mathematical validation

Reviewed 2026-11-22. This document is a research review, not a trained PCMV result. No weights were downloaded, training launched, or submission made by this review. Empirical zero-training measurements, if produced separately, must retain their own protocol and provenance.

**Verdict: a reasonable hypothesis for a bounded zero-training diagnostic, but not a validated route to 0.80.** Decoupled region toggles can bypass saturated pixel logits. The proposed explanation of why counterfactual/JND features must identify true holes is not established, several statistical statements need correction, and the required complete OOF inputs do not currently exist merely by reusing the cited checkpoints. Training should wait for an actual coverage-and-signal result under the corrected controls below.

## 1. Baseline and provenance must be frozen correctly

The requested baseline **#3005 has public mIoU 0.6597** and changes only StegaStamp relative to #2675. Its StegaStamp development OOF is 0.724414 on 170 images. A newer recorded result exists: **#3054, public 0.6677**, with StegaStamp development OOF 0.744788 after the declared 450-step endpoint. Keep #3005 as the requested experimental reference; report #3054 as the newer existing system rather than silently mixing its SS maps into #3005. Neither public result is a pooled local-CV score. [#3005 report](../next_ideas/SUBMISSION_REPORT.md), [latest report and receipts](../next_ideas/REPORT.md), [#3054 report](../next_ideas/SUBMISSION_450_REPORT.md).

A full current 510-image “#3005 OOF” is not already established. PS #2675/#3005 continuation has a **57-image fixed-fold screen**, while the available 170-image PS OOF is the earlier #2409 recipe. The previous 510-image error-budget calculation explicitly mixed protocols. A zero-training evaluation can use the 397 images covered by the current PS/TM/SS validation artifacts, or label a 510-image historical proxy honestly; it cannot call that proxy current #3005 OOF. [Protocol warning](../next_ideas/REPORT.md).

TriForensic's cited local branch also comes from a **fixed fold1**: 113 PS real training images and 57 held-out images. TM was stopped before its gate; this experiment did not train an SS TriForensic branch. Reusing one branch on all 170 images would put its own training images into the purported OOF local-evidence evaluation. Full three-fold PCMV requires fold-specific, correctly excluded local/base weights, or a declared synthetic-only/frozen alternative. [TriForensic protocol](../triforensic_wm/PROTOCOL.md), [report](../triforensic_wm/REPORT.md), [training initialization](../triforensic_wm/train.py).

The 0.735 local AUC and −5.30/+0.009 fusion evidence supports testing an independent decision mechanism. It does not prove that fusion is the sole bottleneck: the local stream also supplied detached learned skips to the base, and the reported local AUC is not region-level specificity at the acceptance rate needed here. [Measured fusion analysis](../triforensic_wm/REPORT.md).

## 2. Counterfactual masking substantially overlaps an existing diagnostic

`next_ideas/oracle_ladder.py` already measured full-image decoding, exact GT bbox decoding, outside-GT blur, outside-GT mean fill, GT ablation, continuous signed soft agreement, and wrong-message references. It explicitly warns that replacements change context and create boundaries. At the old 90% gate, outside-GT mean/blur recovered zero small cases for all three methods; GT crops recovered 3/43 small PS, 21/45 TM and 8/45 SS cases. These are **oracle diagnostic views**, not deployable proposals. They do not test the new calibrated continuous rule, so they neither establish nor refute PCMV. Reanalyze their cached continuous evidence before repeating equivalent decoder calls. [Code](../next_ideas/oracle_ladder.py), [measured results](../next_ideas/REPORT.md#a2-oracle-ladder).

The genuinely new combination is deployable proposal coverage, JND-matched region contrast, calibrated acceptance, and bidirectional mask composition. Background neutralization itself is not new evidence.

## 3. “In-place” is not automatically “1:1 unresized decoding”

The actual official wrappers resize **PS to256, TM to245, SS to400**, including when their input is a 512-pixel neutralized canvas. Preserving a region's location in that canvas preserves its global geometric relation before the common native resize; it does not preserve one input pixel per decoder pixel. A tight crop undergoes a different scaling. Directly feeding 512 into the PS/TM network can be a useful separately declared view, but bypasses the official wrapper and changes the inference distribution. SS also has its spatial transformer, whose estimated warp can change under masking. [Actual wrappers](../../wmloc/watermarks.py).

PixelSeal's own paper describes encoder/downsample/upsample/JND processing and explicitly discusses extraction distribution shifts caused by resizing. Thus the native-resolution choice must be measured, not described as a universal physical guarantee. [PixelSeal, §4.3](https://arxiv.org/html/2512.16874v1).

For linear GAP heads, weighted pooling is algebraically valid: if `h(x)` is the pre-GAP map and `sum_x a(x)=1`, then `W sum_x a(x)h(x)+b = sum_x a(x)(Wh(x)+b)`. The full-mask weighted readout must reproduce ordinary logits numerically before any proposal test. However, masking the pooling support **does not remove background information already inside each feature's receptive field**. PS additionally uses ConvNeXtV2 global-response normalization, which aggregates spatial norms. Small support can collapse to only a few coarse feature cells. [Wrapper per-position logits](../../wmloc/watermarks.py), [verified PS architecture](../triforensic_wm/PROTOCOL.md), [official GRN implementation](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py).

## 4. The normalized alignment is not an exact Gaussian permutation test

Let `v=tanh(logits-clean_bias)` be fixed and let the null message signs be **independent unbiased Rademacher variables**. Then, by direct calculation,

`E[v·s]=0`, `Var[v·s]=||v||²`, so `(v·s)/||v||` has exactly zero mean and unit variance when `||v||>0`.

This does **not** make its finite-sample distribution exactly normal. For one dominant nonzero component it takes essentially only two values. Correlated decoder coordinates alone do not invalidate this conditional random-message variance, because the randomness here is in independently drawn message signs, not in the fixed decoder vector. Problems arise from a mismatched payload null, dependence on the true message during proposal/search selection, and converting a standardized score into a Gaussian tail without checking calibration.

If “permutation” means shuffling a fixed message, its number of ones is fixed. With `B` bits and mean sign `mu`, the correct moments are:

- `E[v·s_perm] = mu sum(v)`;
- `Var[v·s_perm] = (1-mu²)/(B-1) * (B sum(v²) - sum(v)²)`.

Therefore the proposed denominator is generally not the exact standardized fixed-weight permutation statistic. Use an explicitly defined random-message distribution matching the task, handle zero norm with abstention, and estimate the empirical null of the actual search statistic. The PixelSeal paper itself states a uniform-message assumption for its binomial detection bound. [PixelSeal §3.1](https://arxiv.org/html/2512.16874v1).

For hard-bit agreement, the illustrative 256-bit arithmetic is approximately correct, but easily overinterpreted. Calculated using `scipy.stats.binom.sf(k-1,256,.5)`:

| Correct bits | Agreement | Standardized count | One-sided binomial tail |
|---|---:|---:|---:|
|148|57.8125%|2.500|0.00731417|
|149|58.2031%|2.625|0.00512640|
|174|67.9688%|5.750|4.5054e-9|

For **80 independent** candidates, the chance of at least one ≥148-bit accidental match is `1-(1-.00731417)^80 = 44.4%`. This is an illustration, not a claim that actual proposals are independent. Correlated proposals and multiple fills/scales still require calibrating the maximum/selected action across the complete search. Use Monte Carlo p-values such as `(1 + exceedances)/(R + 1)`, never zero from a finite null sample. [Phipson & Smyth](https://arxiv.org/abs/1603.05766).

Fit clean-bit bias only on permitted training covers. A single unconditional mean cannot guarantee removal of content-, mask-, fill- or geometry-dependent bias. Wrong messages can be scored against cached logits cheaply when proposals/views are fixed; they need not trigger another decoder call.

## 5. Neutral-fill ablation is a sensitivity probe, not physical removal of only watermark

Replacing a region with its boundary mean removes image content and texture, inserts a boundary, changes global normalization, and can change STN alignment. Hence neither `Delta L<=0 ⇒ clean` nor `Delta L>0 ⇒ watermarked` is a valid logical implication. A truly marked region may hurt the decoder; a clean region may provide useful context. The subtraction measures decoder response to an artificial intervention, not the causal effect of deleting watermark while holding content fixed. Perturbation-based attribution literature independently documents this out-of-distribution failure mode. [Feature Perturbation Augmentation](https://arxiv.org/abs/2303.01538).

Measure several fixed fill controls, record original versus changed STN behavior where applicable, and compare the **same proposal on exact clean/full-watermarked twins**. A useful extra diagnostic is a difference-in-differences between the marked and clean twin's ablation response. That controls some content and boundary effects, but remains a synthetic diagnostic rather than proof of real counterfactual validity.

## 6. JND is useful conditioning, not known statistical capacity

PixelSeal JND is a perceptual attenuation map computed from luminance adaptation and contrast masking. It is not a calibrated probability of detection, residual energy, or Fisher information. On an edited image it is also not necessarily the embedder's original map. Its use as “capacity” for TM/SS is an additional hypothesis, since those families do not share this exact attenuation mechanism. [Official JND code](https://github.com/facebookresearch/videoseal/blob/main/videoseal/modules/jnd.py), [PixelSeal model card](https://github.com/facebookresearch/videoseal/blob/main/videoseal/cards/pixelseal.yaml).

The cited 51.45% unchanged RGB pixels and 74.88% in low-JND regions came from **16 synthetic embed64 covers**, not a universal audit of real PS images. Zero residual at an individual pixel proves that label provenance is not the same as a nonzero pixel change. It does not establish absence of informative neighboring/contextual signal, a region-level ambiguity rate, or an information-theoretic ceiling. [Original bounded measurement](../small_zoom/RESEARCH_BRIEF_AFTER_2409.md).

`Delta L / sum(JND)` is a heuristic: the numerator is bounded and nonlinear, while the denominator depends on area and units. Near-zero capacity amplifies numerical noise; large areas are mechanically penalized. Retain raw drop, area and JND summaries, impose a predeclared low-power abstention rule, and calibrate against matched nulls instead of interpreting the ratio's sign as a physical classifier.

Use common absolute/train-fitted JND bins or explicit matching, not separate within-S and within-ring quartiles that may represent different actual texture levels. A ring of equal area is impossible when S occupies more than half the image; define capped support and missingness. A surrounding ring need not have opposite provenance. Adjacent forensic-map values overlap receptive fields: ordinary Welch t p-values based on independent pixels are invalid. Use the t-like quantity only as a feature unless calibrated with spatial/block or image-level resampling. The existing carrier has RF29, so thin supports can contain no uncontaminated interior at all. [Carrier and locality tests](../triforensic_wm/model.py), [RF29 caveat](../triforensic_wm/REPORT.md).

## 7. Proposal coverage and shortcut risks remain empirical

SAM2 is a promptable image/video segmenter; it does not promise masks for all edit provenance, absent/inpainted objects, thin rectangles or disconnected remnants. Its automatic generator samples points/crops, filters predicted mask quality/stability and applies NMS. Freezing it does not turn the masks into edit labels. Proposal oracle coverage is therefore the right first diagnostic. [SAM2 paper](https://arxiv.org/abs/2408.00714), [official automatic generator](https://github.com/facebookresearch/sam2/blob/main/sam2/automatic_mask_generator.py).

The dataset's rectangular/fragment geometry categories are proxies, not definitive edit-type annotations. A mask may contain several components; an edit's original rectangular constraints need not constrain a surviving fragment's bounding box. No RGB/semantic embedding in the scorer reduces one shortcut, but proposal membership, area, boundary alignment and salience-related ranks still encode object-selection priors. Same-image unedited hard negatives and source-grouped splits remain essential. [Dataset error audit](../next_ideas/REPORT.md).

For Stage0, generate/cache proposals without GT; use GT only afterward to label exact action gains and compute oracle coverage. Deduplication/capping must also be GT-free. Evaluate the deployable ≤2-action policy, not an unconstrained oracle union of dozens of masks. State whether actions occur on the official128 grid or are composed at512 then majority-downsampled; these operations need not commute.

## 8. Two disjoint positive gains can combine into a negative gain

Official two-class mIoU is a sum of ratios. **Disjoint proposal support does not make action gains additive.** A small computed counterexample:

```python
y =    [1,0,0,0,1,0,0,0,1,1,0,1,0,1,0,0,0,1,1,1]
base = [1,1,0,0,0,0,1,1,1,0,0,1,0,0,1,1,1,0,0,1]
carve_A = [0,6,7,14,2,9,11,10]
carve_B = [16,19,15,18,1,4,8]
# Both actions set their disjoint supports to zero.
# mIoU(base) = 0.2895833333333333
# mIoU(A only) = mIoU(B only) = 0.3055555555555555
# mIoU(A and B) = 0.275
```

These are algebraic test data, not competition measurements. Spatially permuting cells can make the two supports contiguous without changing the scores. Thus training both gain heads only relative to `M_base` and applying two stale positive scores is not safe. Recompute conditioned marginal gains after the first action, predict the joint pair outcome, or restrict the initial screen to one action. For K≤80, Stage0 can enumerate all compatible pairs exactly rather than treat a greedy procedure as the true two-toggle oracle. [Official score definition](../../wmloc/utils.py).

## 9. OOF and anti-shortcut gates need precise definitions

- Split by image/source, never proposal row. All proposals, clean/full twins and alternative realizations of one source stay together. Thousands of proposals from113 images are not thousands of independent images.
- Within an outer fold, fit the bit-bias estimate, feature normalization, scorer, thresholds and hyperparameters using only that outer training set. Meta-training predictions should themselves be generated by inner cross-fitting if base/local models were trained on real labels. A global OOF table is insufficient if its training-row predictions were made by models that saw the current outer validation labels. [Stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html), [nested-CV documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [Cawley & Talbot](https://jmlr.org/papers/v11/cawley10a.html).
- For a **verifier-only wrong-message diagnostic**, freeze base masks and proposal sets and swap only the new message probes. Report treatment-minus-Control-B utility. Geometry/local-only gains need not disappear under message swap. For an **end-to-end null**, rerun all message-dependent proposal/search/selection steps. These answer different questions. A bitwise complement alone is not a sample of the random-message null.
- Define false toggles as **harmful changes**, e.g. negative actual mIoU delta relative to the base, and additionally count wrong-polarity pixels. A clean twin may have a false-positive base mask; correctly carving it is not a failure. Full-WM twins can likewise benefit from additions. Count no-op actions separately. “Invariance” should mean correct provenance behavior, not identical outputs on twins whose true labels differ.
- Observed FPR<2% is not a confidence guarantee. On200 independent images, zero failures gives a one-sided exact95% upper bound1.4867%; one failure gives2.3498%; three failures gives3.8310%. A low-JND quartile of50 images with zero errors cannot establish FPR≤2% at95% confidence. Report counts and intervals, not only pass/fail.
- On a perfectly predicted all-clean image, adding even one false-positive cell changes absent-class IoU from1 to0, causing about0.5 mIoU loss. Thus the mean-loss<0.003 twin gate can be much stricter than a2% harmful-toggle rate. Use the actual official empty-union convention consistently. [Score implementation](../../wmloc/utils.py).

Proposal AUC should compare correct versus wrong **action gains** within images, with minimum-size/positive-gain definitions fixed before inspection. Pixel-label purity and edit status are not the same as expected improvement over an already good base. Report AP/precision and harmful-toggle rate at the actual acceptance rule; AUC0.80 alone does not guarantee safe selection of the best of80 proposals. Bootstrap by image/source, not by proposal, and label intervals conditional on the already-trained models.

## 10. Corrected decision and compute scope

The zero-training sequence should be: provenance inventory → cached-oracle continuous reanalysis → GT-free proposal coverage with exact one/two-action oracle → feature separation with matched content/fill/wrong-message controls. Only then consider scorer training. The currently available local branch supports a valid fixed-fold diagnostic, not a claim of full three-family OOF.

The proposed +0.12 oracle and AUC gates are engineering targets, not derivations of an attainable0.80 score. A negative result can justify stopping **this verifier, proposal pool and compute budget**. It cannot prove that stronger local residual features are the unique bottleneck; low proposal recall, inappropriate nulls, changed decoder context, coarse spatial features, wrong embedding geometry and insufficient power remain alternatives.

Timing needs a measured pilot on the actual device. At510 images,80 proposals and four decoder views, the direct count is163,200 decoder-image evaluations before additional scales/fills. At6,000–9,000 synthetic images,50–80 proposals and four views, it is1.2–2.88 million evaluations. Wrong-message logits can usually be reused; automatic SAM2 mask generation may evaluate many more raw prompts/crops than the final retained K. Neither10 minutes for SAM2 nor10–18 GPU-hours overall is established for this repository or the RTX3070.

**Recommended disposition:** retain PCMV as one coherent research hypothesis, correct the statistical/provenance/composition issues, and demand positive zero-training evidence before spending on the proposed trained scorer. Existing diagnoses motivate the experiment; they do not validate it in advance.
