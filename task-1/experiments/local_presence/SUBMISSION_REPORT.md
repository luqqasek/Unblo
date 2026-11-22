# Submission #2675 — new best public result

Accepted on 2026-11-22: **public mIoU 0.6440**, previous best #2409 **0.6369**.
The server reports delta **+0.007**; displayed scores are rounded. The final hidden
competition score and per-family public scores are unknown. One frozen candidate
was submitted once; no leaderboard-driven threshold/checkpoint search was used.

PixelSeal: continue the previous full170-image model for150 updates with12synthetic
+4real views, LR2e-6 encoder/2e-5 new modules, threshold0.5. Selection evidence is
one reused57-image fold:0.557485→0.570883. This is not full OOF for the new model.
Tiny foreground IoU remains0; the gain mainly concerns larger regions.

TrustMark: keep all old nonempty masks and scan only38 empty test predictions.
The same frozen crop grid/refinement as validation accepts8 new rectangles under
an integer90/100 raw-bit agreement gate. Wrong-query controls accept0/38.
A float32 boundary bug was corrected before upload:90/100 was represented as
0.899999976. This restores the declared threshold; no validation case lay on that
boundary, so reported validation metrics remain unchanged. Worker logs count7
under the old float comparison; the authoritative assembly uses8 under exact
integer verification. No test labels were used for the correction.

StegaStamp masks are identical to #2409. Changed test images:279 PixelSeal,
8 TrustMark,0 StegaStamp. This combined submission does not isolate the public
gain by family or by component.

Artifact:1020unique names, binary uint8[1020,128,128],16,734,774bytes.
SHA256: `67b1a62d4d47589095e1b05d8e7471271ede3576c3130a0f1c1c96904ddf76a4`.
File: `results/submission/submission-local-presence.pt`.
Raw server receipt: `results/submission/server_receipt.json`.
Dashboard receipt: `results/submission/upload_receipt.json`.
Checks and sources: `results/submission/assembly_validation.json`.

Production took7m54s on one4×A10040GB node. All five experiment/production jobs
completed; combined allocated node time25m42s. Caches6.44GB were removed, selected
models and prior submissions retained. The node was released.

For reproduction, see PLAN.md and report commands; `job5` performs the fixed
full-data continuation/test inference, `assemble` combines masks, and the existing
upload_submission utility sends one validated artifact. The API key is read from
stdin and is absent from source, artifacts and receipts.
