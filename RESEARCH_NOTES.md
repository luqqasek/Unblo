# What we tried and what we kept

## Task 1

The final 0.7309 solution combines the frozen two-member DINO recipe with a PixelSeal specialist refined using feature CutMix. It supersedes earlier family-specific submissions, including 3485/public 0.6777. Expected-mIoU decoding chooses a binary region size from the probability ordering rather than using only a fixed 0.5 threshold.

The final PixelSeal edit changed 191,745 decoder parameters. It scored 0.630891 against 0.629340 on 57 reused development images. Its paired interval includes zero and the promotion gate failed. The user then requested an actual verification upload, which scored 0.7309 publicly. Other-family masks also differ from the previous four-member public ensemble, so this public gain cannot isolate the PixelSeal edit.

Rejected approaches included native small-region recognition, PCMV refinements, clean-hole adaptation, direct output-head scaling/refits and structural weight merging. Their bounded failures do not prove a fundamental ceiling. The attempted 0.80 target was not reached. Full chronological research remains in the upstream history and archived dashboard; the submission view keeps the selected source and evidence.

## Task 2

Pure behavioral similarity confused disguises with changed weights. The selected rule first checks adapter and quantized-copy likelihoods, then groups detected disguises with the base model. The final uploaded CSV scored 1.0 publicly after earlier candidates scored 0.55 and 0.30.

Family assignments and disguise tags in the submitted pipeline include manual decisions. The `ep-26` exclusion explicitly uses feedback from an earlier public submission. The result therefore is not an untouched, independently evaluated model-selection experiment. Later Qwen-3B evidence must not be silently substituted when reproducing the historical upload. A saved test-bed result is substantially lower than the public result.

## Task 3

Early token-drop consensus distillation produced nearly unchanged predictions and a public score of 0.0000 for CTDC v1. Head-bias class suppression also scored zero. Neither is the selected solution.

Per-image target distributions improved the public score to 0.2258, then stronger preservation and an altered-view shift led to 0.2994. Adding CLS-feature preservation produced E30's 0.3013. A later conflict-projection candidate E36 scored 0.2922, so E30 remained the reference. A later 16-candidate batch raised normalized CLS preservation to 10, giving E50 **0.3149**, the current selected checkpoint. Synthetic-bank and drift metrics did not consistently rank public performance. Gold-model matching and true retain accuracy remain hidden.

The exploratory code, chronological handoffs, queued launchers, duplicated dashboards and large intermediate outputs were removed from the submission view. Task 1's earlier source remains in Git history. Task 2's original work remains at its upstream revision. The separate Task 3 checkout and local-only experiments were backed up before consolidation.
