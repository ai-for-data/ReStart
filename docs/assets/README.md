# Release illustrations

Three original looping animations explain the method. They play directly in Markdown without JavaScript, external hosts, or a demo service.

| Process | Animation | Static alternative |
|---|---|---|
| Expert context → conditional prediction → native action sampling | [GIF](restart-context.gif) | [PNG](restart-context.png) |
| Stepwise inverse solve → native forward replay → admission | [GIF](restart-inversion.gif) | [PNG](restart-inversion.png) |
| SCM graph → supervised table → prior pretraining | [GIF](restart-scm.gif) | [PNG](restart-scm.png) |

## Render

From the repository root:

```bash
python -m pip install -r requirements-docs.txt
python scripts/render_release_assets.py
```

Install TeX Live with `pdflatex` and DejaVu Sans first. The renderer can also locate DejaVu Sans through an installed Matplotlib package. It compiles all mathematical formulas with LaTeX, rasterizes the transparent equation plates, and composites the animated diagrams. Generated TeX and equation plates are retained in [`equations/`](equations/).

Each GIF is 1440 × 820 pixels and loops over 32 seconds, with pauses between dependent stages. Still images provide a reduced-motion alternative.

## Reading the animation

- The context animation keeps one fixed sampling map. Expert actions yield paired support rows; the complete support and current query arrive before prediction. The predicted noise combines with the native history anchor before action generation. Gold/coral denotes expert support, blue denotes current conditions/history, and mint denotes prediction or forward replay.
- The inverse animation begins with a demonstrated action, walks backward one step at a time, then replays forward and assembles a checked situation–start pair. The drawn steps illustrate a chain; they do not change a model's inference recipe.
- The SCM animation computes a toy causal rule, adds numeric support rows, hides the next answer, and reveals it as pretraining supervision. These are calculated toy values, not outputs of a running predictor. The animation does not infer a robot causal graph or pretrain on robot demonstrations.

Paths, table values, highlights, and correction iterates are schematics. They are not measured trajectories, convergence evidence, or extra benchmark results. No model, robot data, network access, or GPU is used by the renderer. [Technical attribution](../credits.md).
