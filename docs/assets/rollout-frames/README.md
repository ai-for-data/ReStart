# Manuscript rollout frames

These 32 frames reproduce the displayed image panels of the manuscript's demonstrations-and-rollouts figure, preserving its orientation, task order, and within-rollout order. No robot scene content was synthesized or retouched.

Each task has one clean expert demonstration frame, three clean rollout frames, and four randomized rollout frames. Each rollout is the first successful episode of its condition. These qualitative examples are not aggregate performance estimates.

Run `python scripts/render_rollout_examples.py` from the repository root to rebuild the dark presentation layout (Matplotlib and Pillow required).
