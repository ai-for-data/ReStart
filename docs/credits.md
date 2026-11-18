# Technical provenance and third-party materials

The release presentation uses method-level names. This file preserves implementation provenance; it does not claim that ReStart developed its upstream components.

- Frozen robot-policy integration: [UniWAM](https://github.com/UniWAM/UniWAM).
- Released context predictor: [TabICLv2](https://github.com/soda-inria/tabicl). The SCM background illustration summarizes [§5 of its paper](https://arxiv.org/html/2602.11139v1#S5), including random root propagation and column extraction. It does not add Gaussian noise at every node to the default v2 recipe.
- Simulation benchmark: [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin).

Exact source snapshots are recorded in [`upstream-sources.lock.json`](../upstream-sources.lock.json). Code, checkpoints, and datasets retain their respective licenses. This repository does not grant a new license for those materials.

Presentation references include [Hugging Face Daily Papers](https://huggingface.co/papers/trending) and the official project repositories above. The release animations are original code-rendered illustrations. The rollout examples reproduce the manuscript’s RoboTwin scene frames; the underlying simulator assets retain their original terms.

Public documentation is not a patent filing or a confidentiality mechanism. The included source remains inspectable.
