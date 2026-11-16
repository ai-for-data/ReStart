# ReStart

### Redirect, Don't Rewrite

**Adapt a frozen world-action model through its sampling start point.**

ReStart turns expert demonstrations into an inference-time context. It inverts demonstrated actions into starting noise, predicts a start for the current situation, and lets the frozen policy generate the action through its existing sampling process.

![ReStart: expert context, conditional prediction, native action sampling](docs/assets/restart-context.gif)

[How it works](docs/method.md) · [Results](docs/results.md) · [Integration](docs/reproduction.md) · [Static illustrations](docs/assets/README.md)

## Context adaptation without policy updates

1. **Invert expert actions.** Recover starting points by stepwise inversion and verify them by forward replay.
2. **Build context.** Pair the current-condition representation with the admitted inverse noise at the native replanning times throughout expert trajectories.
3. **Predict and sample.** Query the fixed context predictor, combine its noise prediction with the policy's native history anchor, and run the original action sampler.

The frozen policy retains its action representation, history conditioning, and sampling schedule. ReStart changes the predicted noise component of its starting point.

![Expert action inversion](docs/assets/restart-inversion.gif)

## Results on RoboTwin 2.0

The released Action-only evaluation covers **50 tasks × 2 conditions × 100 episodes**.

| Condition | ReStart | Authors' reported fine-tuned policy |
|---|---:|---:|
| Clean | 80.74% | 75.14% |
| Randomized | 80.82% | 68.32% |

The fine-tuned reference is taken from the authors' report, not a reproduction on our evaluation stack. See [per-task counts](results/robotwin-50.csv) and [comparison scope](docs/results.md). The separate frozen-Base control covers **19 tasks**; its findings do not establish a full-benchmark Base comparison.

## Learning to use a context

The context predictor is pretrained on synthetic supervised tasks. At robot adaptation time, it uses expert-derived examples without updating the policy's parameters.

![Synthetic-prior pretraining](docs/assets/restart-scm.gif)

These animations illustrate the method; they are not recorded robot rollouts or measured convergence traces.

## Code and reproduction

Start with the [integration guide](docs/reproduction.md). This repository contains the adaptation modules, native policy integration, support-library construction, result tables, and illustration source. External policy weights, robot assets, and the upstream simulator are obtained separately from their providers.

[Technical provenance](docs/credits.md) · [Source snapshots](upstream-sources.lock.json) · [Usage notice](NOTICE.md)
