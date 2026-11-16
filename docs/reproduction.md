# Native-policy integration

This release provides the adaptation code and its integration contract. It does not bundle model weights, demonstrations, simulator assets, or a turnkey cloud deployment.

## Environment and sources

Use the official UniWAM and RoboTwin installation instructions for the policy and simulator. Fetch the exact source revisions in [`upstream-sources.lock.json`](../upstream-sources.lock.json). The source paths and declared checkpoint identity are recorded in [`benchmarks/policy-assets.json`](../benchmarks/policy-assets.json); place these upstream checkouts under the corresponding `.deps/` paths relative to this repository.

Install this repository into the native policy environment with Python 3.10 or newer:

```bash
python -m pip install -e .
```

Install the context predictor into its own compatible environment:

```bash
python -m pip install -e '.[pfn]'
```

The predictor's interpreter, checkpoint, and devices are passed explicitly to the adapter. The policy environment must retain its official PyTorch and simulator dependencies. See [third-party provenance](credits.md).

## Data and labels

[`robotwin_experts.py`](../embodpfn/robotwin_experts.py) constructs expert bundles at native replanning times and preserves the measured history. Inspect its CLI with:

```bash
python -m embodpfn.robotwin_experts --help
```

From the native RoboTwin working directory, the offline support entry is:

```bash
python /path/to/ReStart-official/scripts/robotwin_produce_support.py --arguments /path/to/arguments.json
```

`arguments.json` must contain the arguments required by the original UniWAM loader and the explicit adapter arguments below. Use the original deployment configuration and the values from its original evaluation recipe; do not substitute a shortened evaluation or a new sampling schedule.

| Adapter argument | Purpose |
|---|---|
| `embodpfn_support` | Persistent support-library file; production refuses to overwrite an existing file |
| `embodpfn_expert_bundle` | Expert bundle containing its manifest, observations, actions, and measured history |
| `embodpfn_inverse_reference` | Reference manifest binding the inversion sources, native recipe, and checkpoint profile |
| `embodpfn_noise_target` | `action` for the reported Action-only results; `joint` selects the joint backend |
| `embodpfn_policy_receipt` | Checkpoint provenance receipt, when required by the declared checkpoint path |
| `embodpfn_pfn_python` | Interpreter for the context predictor |
| `embodpfn_pfn_checkpoint` | Predictor checkpoint |
| `embodpfn_pfn_device` / `embodpfn_pfn_devices` | Explicit predictor device placement |
| `embodpfn_pfn_cache` | Supported native cache mode: `none`, `kv`, or `repr` |
| `embodpfn_pfn_execution` | Predictor execution implementation; default `native` |

The reference manifest is validated in [`robotwin_policy.get_model`](../embodpfn/robotwin_policy.py), using [`reference_recipe`](../embodpfn/flowdagger.py), [`robotwin_recipe`](../embodpfn/official_recipe.py), and the checkpoint's `config.json` digest. These checks intentionally require the actual source and assets; this release supplies no fabricated example receipts or substitute benchmark configuration.

## Evaluation

Use RoboTwin's original `script/eval_policy.py` loop and UniWAM's original deployment configuration. The policy plugin in [`robotwin_policy.py`](../embodpfn/robotwin_policy.py) exposes `get_model`, `eval`, and `reset_model`, delegates to the original policy, and attaches the context adapter at action sampling. Register it through the upstream policy-loading mechanism rather than replacing the evaluation loop.

Supply the declared support library and matching manifests for the task. Evaluate the full official episode count and retain the official condition, seed, instructions, and action timing. The native configuration remains the authority for sampling steps and history conditioning.

The public result tables are aggregate evidence, not downloadable support banks. Full reproduction additionally requires the external assets and generation of task-specific support libraries. The lightweight persistence test checks support integrity; it is not a simulator or numerical benchmark certification.
