# Look-at improvement plan: conservative continued fine-tuning

## Evidence checked before changing code

The November 3 experiment uses the fixed nine-source, 10% training manifest
`scene10pct_s42`. The original VSI-Bench test split remains unchanged.

| Check | Observation |
|---|---|
| Base evaluation | 73.538405 overall |
| All evaluation | 73.050616 overall; difference -0.487790 percentage points |
| Coverage | Both contain the same 5,130 unique questions from 288 videos |
| Responses | Neither run has empty responses or invalid first-token number/option formats |
| Adapter identity | Evaluation merged `runs/scene10pct_s42/all`, not the smoke adapter |
| Training completion | Both `cam` and `all` completed 7,458 optimizer updates |
| Average training loss | `cam`: 0.064901; `all`: 0.065652 |
| Learned geometry/visual gates | 0.099820 / 0.099751, versus initialization at 0.1 |
| Numerical health | Saved projector/statistics are finite; no evidence of a failed or randomly reinitialized evaluation projector |

Recomputing the official eight-category macro average from saved per-question
scores reproduces both reported results. A paired bootstrap resampling the same
288 videos for both models, with 5,000 replicates and seed 42, gives a 95%
interval of **[-1.616650, +0.675231] percentage points** for `all - base`.
The observed decrease is therefore not sufficient to establish a population-level
regression, nor does the interval demonstrate equivalence or improvement.

The trained model improves room size, relative distance, and route planning,
but loses more on relative direction, appearance order, and absolute distance.
The numerical and multiple-choice answer formats are valid, so malformed
generation is not the explanation for these losses.

An audit of three deterministic training videos from each of the six video
sources found valid shared geometry and both near/far matches. This is a limited
cache sanity check, not proof that all geometric matches identify the correct
object or that revisits improve accuracy.

## What we can and cannot conclude about the cause

There is **no demonstrated implementation failure explaining the score gap**.
The existing comparison changes continued fine-tuning, VGGT context, and
cross-anchor gathering together. The original pod is currently training `self`;
its completed `cam` model and eventual `self` model remain necessary controls.

The optimization recipe is a reasonable first intervention:

- The released Cambrian-P checkpoint is already spatially fine-tuned. Updating
  its LLM adapters at the same peak rate as a newly initialized projector may
  unnecessarily move a strong baseline while the new input pathway learns.
- `all` has larger pre-clipping gradient spikes than `cam` (maximum logged
  non-projector norm 36.40 versus 3.53). Clipping is already enabled and losses
  remain finite; these spikes are a warning, not proof of instability causing
  the benchmark decrease.
- Both gathering gates remain very close to their initial values. That does
  not prove gathering is unused, but motivates recording routing and gate
  diagnostics during ordinary training rather than only during the smoke test.

**Hypothesis:** smaller pretrained-LLM updates, while retaining the projector's
learning rate, will better preserve existing capabilities and give the added
pathway relatively more opportunity to adapt. This is a controlled optimization
experiment, not a promised accuracy fix or a demonstrated root cause.

Do not redesign the matcher, change the radius, zero-initialize the appended
token, increase epochs/data, or select checkpoints using VSI-Bench at this stage.
In particular, the existing nonzero projector initialization avoids the
previously observed near-zero-token RMSNorm gradient amplification.

## Planned implementation

1. Keep all existing v1 stages and defaults unchanged.
2. Add explicitly named `cam_v2`, `self_v2`, and `all_v2` arms:

   | Setting | Existing v1 | Conservative v2 |
   |---|---:|---:|
   | LLM LoRA peak learning rate | 1e-4 | 1e-5 |
   | VGGT projector peak learning rate | 1e-4 | 1e-4 |
   | Epochs / seed | 1 / 42 | unchanged |
   | GPUs / per-device batch / accumulation | 2 / 1 / 4 | unchanged |
   | LoRA rank / alpha | 64 / 128 | unchanged |
   | Anchors / added VGGT tokens for self/all | 128 / 128 | unchanged |
   | Training manifest, caches, normalization | fixed existing files | unchanged |
   | Architecture, gathering and initialization | existing implementation | unchanged |

   Each arm starts from the released base checkpoint in a separate output
   directory. It must not resume the old `all` run or overwrite its results.
3. Log the latest observed video gathering statistics in each logging interval:
   near/all neighbor coverage, distant-neighbor coverage, and the two gates.
   Clear consumed statistics so image-only intervals do not reuse old values.
   These are rank-local, latest-video diagnostics, not global dataset averages.
4. Correct full-evaluation verification to support the actual
   `vsibench.json` object containing `logs`, while retaining JSONL support.
   Both formats must enforce expected unique question count, nonempty
   generations, and finite metrics. Allow checking selected completed arms.
5. Propagate evaluation-launch failures instead of printing failure and
   returning success.

## Validation and deployment

- Add focused regressions for original/v2 launch arguments, both sample-file
  formats, invalid evaluation outputs, failure propagation, and consumed
  gathering diagnostics.
- Run the relevant CPU regressions, existing formatting checks for the edited
  test files, shell syntax checks, and whitespace checks before committing.
- Commit the plan, code, and tests locally, then push the experiment branch.
- Both pods mount the same `/data` PVC, including the original checkout and
  venv. Use a separate Git clone at
  `/data/lz1bhc/cambrian-p-lookat-v2` on `1001200-lz1bhc-2gpu-b200-2`.
  Do not pull into the original checkout, reinstall shared packages, edit
  existing caches, or interrupt the other pod's `self` job.
- Verify imports resolve to the isolated checkout through `PYTHONPATH`.
  Reuse the existing managed-Python environment and cached model dependencies.
- On the free pod, launch `train-all-v2` persistently on its two B200s and
  verify real optimizer updates, finite losses, correct learning rates, and
  growing persistent logs before reporting that training is running.

The expected output directory is
`/data/lz1bhc/runs/scene10pct_s42/all_v2`; the training log is
`all_v2/train.log`. The corresponding commands are:

```bash
bash scripts/vggt17/run_b200.sh train-all-v2
bash scripts/vggt17/run_b200.sh eval-all-v2
bash scripts/vggt17/run_b200.sh check-evals base all all_v2
```

Only training is to be started automatically now. Evaluation remains a full,
unchanged benchmark run after completion.

## Interpreting the next result

Finish the existing v1 controls: `cam - base` measures continued fine-tuning,
`self - cam` measures the added VGGT pathway, and `all - self` measures gathering
under the original recipe.

`all_v2 - all` measures the learning-rate intervention, not gathering's causal
contribution. A gathering claim under v2 needs independently trained `self_v2`
and `all_v2` with the same conservative recipe; `cam_v2` supplies the continued
fine-tuning control. Do not compare `all_v2` against v1 `self` as if they were
matched controls.

Keep the benchmark questions and scoring untouched. Do not turn repeated
benchmark evaluations into hyperparameter or best-checkpoint selection.
