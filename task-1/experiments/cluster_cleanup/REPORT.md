# Cluster cleanup — 2026-11-22

Completed under explicit user authorization. **47,680,048,490 logical bytes recovered (47.68 GB / 44.41 GiB)**, including log compression. Before cleanup the project occupied approximately88GiB (`du`, rounded). This is project-file accounting, not a claim about all users' filesystem free space.

- Removed redundant COCO archive:19,336,861,798 bytes.
- Removed577 download chunks:19,336,861,798 bytes.
- Removed four rejected PS/TM native RGB checkpoints:6,004,127,848 bytes.
- Removed debug overfit StegaStamp checkpoint:1,501,031,706 bytes.
- Removed non-selected StegaStamp synthetic-dev `best.pt`:1,501,031,962 bytes. The deployed raw `last.pt` is preserved.
- Compressed30 scheduler stdout/stderr logs losslessly, verifying decompressed hashes; recovered133,378 bytes.

Preserved all11 production/synthetic-base/OOF checkpoints listed with SHA256 in [completed.json](completed.json). Preserved decoded COCO photos, source partitions, instance annotations, encoded clean/marked PNGs, official embedder weights, environments, metrics, predictions and recipes. Old legacy checkpoints on Windows were not touched. No active Slurm jobs were present when deletion began.

Safety checks: all118287 expected COCO image names were present in the extracted directory; five deterministic sample images matched archive size+CRC. This is not an exhaustive checksum audit of every extracted pixel. Removed files were regular non-symlink files under the fixed project root, with no hard links. Deleted weights were not needed by either accepted submission. Final checks confirmed deleted paths absent and all protected checkpoints present with original size. Hashes were recorded before deletion; protected files were not modified.

[Deletion plan](plan.json) · [Executed receipt](completed.json) · [Guarded cleanup script](cleanup.py).
The script is one-time and refuses to run again after completion. Do not rerun a historical deletion plan against newly trained weights.
