# MV-CoL: Multi-View Contrastive Learning

This repository provides a reference implementation of the MV-CoL framework described in the paper. Dataset-specific tuned configurations, trained checkpoints, fixed experimental splits, and extracted representations are not included. Therefore, exact numerical reproduction of the results reported in the paper is not guaranteed.

The implementation is complete and runnable; the lack of exact numerical
reproduction comes from omitted experimental assets, not from disabled code or a
changed algorithm.

## Method overview

```text
View A: definition-guided prompt ─┐
View B: behavior-cue prompt ──────┼─ shared Meta-Llama-3-8B-Instruct encoder
View C: holistic-judgment prompt ─┘  + shared classification head
                                      │
                         per-view CE + SupCon + Triplet
                                      │
                 raw last-position h_A, h_B, h_C extraction
                                      │
       originals + absolute differences + products + mean fusion
                                      │
       fold-local StandardScaler → PCA → Borderline-SMOTE
                                      │
                Optuna-selected stacking ensemble
                                      │
                   independent test evaluation
```

The three view templates are defined in:

- `prompts/definition_view.py`: View A, with class definitions and criteria.
- `prompts/behavior_view.py`: View B, emphasizing local behavioral cues.
- `prompts/holistic_view.py`: View C, judging the complete post.

They contain no task-specific labels. At runtime, all three obtain the dataset
name, task description/instruction, label names, label definitions, behavior
cues, and allowed output labels from the selected YAML config through
`prompts/common.py`. This supports all six paper tasks without an ICAP-only
Python prompt:

- `configs/question.yaml`
- `configs/answer.yaml`
- `configs/opinion.yaml`
- `configs/urgency.yaml`
- `configs/icap_psy.yaml`
- `configs/coi_ast.yaml`

The task files inherit the executable, explicitly non-final hyperparameters in
`configs/reference_defaults.yaml`. Source datasets using numeric labels can use
the task-specific `label_aliases` mapping; unknown labels fail validation.

All views are forwarded separately through the same Llama encoder and the same
sequence-classification head. They are not treated as independent samples in one
contrastive batch.

## Stage one: 4-bit LoRA and per-view metric learning

The default backbone is `meta-llama/Meta-Llama-3-8B-Instruct`, loaded with NF4
4-bit quantization and adapted with LoRA. Rank, alpha, dropout, and target modules
are visible in `configs/reference_defaults.yaml`.

For each view independently:

1. The final valid-token hidden state passes through that view's projection head.
2. SupCon L2-normalizes projected features, uses same-label samples as positives,
   uses all other non-anchor samples in the denominator, excludes the anchor, and
   applies the configured temperature.
3. Triplet loss uses batch-hard mining within that view: the farthest same-label
   positive and nearest different-label negative optimize
   `max(0, D(a,p) - D(a,n) + margin)`.
4. CE is computed from the shared classification head.

The projection-head wording in the paper can be read ambiguously. This released
reference implementation preserves its current behavior explicitly with
`projection_head_mode: separate`: each view has its own metric projection head,
while the Llama encoder and classification head remain shared. The code also
accepts `shared` only as an explicit opt-in experiment; it is not the default and
is not silently selected.

The paper specifies the triplet-loss formula but does not prescribe a mining
rule. The released implementation therefore records its concrete choice as
`triplet_mining: batch_hard`: farthest same-label positive and closest
different-label negative within each view. This strategy is retained unchanged.

The CE, SupCon, and triplet values are each averaged over Views A/B/C and then
combined as `L = L_ce + lambda_1 L_sup + lambda_2 L_tri`. The CE coefficient is
fixed at one; the shared YAML exposes `lambda_supcon` and `lambda_triplet`.
A batch needs repeated labels and at least two classes for SupCon/triplet anchors
to be active; anchors without a valid positive or negative are safely omitted.

The paper reports selecting LoRA settings, loss weights, temperature, and margin
through validation. This repository exposes those values in YAML so local runs
can compare candidate configurations using Train+Validation, but it does not
embed the paper's selected per-dataset values. The test split must not be used for
that comparison.

## Stage two: representation extraction and fusion

After fine-tuning, the adapter is loaded in inference mode and frozen. For every
user-provided train, validation, and test split, the implementation extracts the
last layer's hidden state at the final non-padding token:

```text
h_A, h_B, h_C
```

These stage-two representations are **not L2-normalized**. Fusion follows the
paper's concatenation order exactly:

```text
[h_A,
 h_B,
 h_C,
 |h_A-h_B|,
 |h_A-h_C|,
 |h_B-h_C|,
 h_A⊙h_B,
 h_A⊙h_C,
 h_B⊙h_C,
 (h_A+h_B+h_C)/3]
```

## Leakage-safe model selection

The candidate pool is XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees,
KNN, and NuSVC. Optuna selects a valid subset and its example configuration;
Logistic Regression is always the stacking meta-classifier.

Fixed estimator counts and all tunable ranges are declared in the shared YAML as
reference compute-budget defaults, not hidden paper-final best parameters.

Each base estimator is an imbalanced-learn Pipeline containing StandardScaler,
PCA, Borderline-SMOTE, and the classifier. Consequently, preprocessing and
resampling are fitted inside each Stacking training fold. Validation and test data
are transformed only and are never used to fit preprocessing or SMOTE.

Optuna maximizes validation Weighted-F1 exclusively. Validation Accuracy is
printed for diagnostics but cannot select the best trial. The example uses 30
trials; this is a minimum demonstration setting and can be increased.

After selection, train and validation are combined and the selected stage-two
Stacking model is refitted. The test feature file is first loaded only after this
final fit, and the test split is evaluated once.

## Data format

Supply independent JSONL files for train, validation, and test. Each row contains
an ID, text, and label:

```json
{"id": "example-001", "text": "I think this follows because ...", "label": "Constructive"}
```

The repository does not create, encode, or distribute fixed sample indices.
Dataset-specific label semantics and optional numeric aliases are declared in the
six task YAML files. See `data/README.md` for the complete input contract.

## Installation and execution

Python 3.10+ and a CUDA GPU supported by bitsandbytes are recommended. The Llama
model is gated; accept its license and authenticate with Hugging Face first.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
cp configs/question.yaml configs/my_run.yaml
```

Choose the appropriate task config (Question is only an example above), then edit
`configs/my_run.yaml` to point to your licensed model/data paths and run:

```bash
python main.py --config configs/my_run.yaml --stage validate
python main.py --config configs/my_run.yaml --stage train-lora
python main.py --config configs/my_run.yaml --stage extract-features
python main.py --config configs/my_run.yaml --stage fuse-features
python main.py --config configs/my_run.yaml --stage train-ensemble
```

The complete reference path is also available:

```bash
python main.py --config configs/my_run.yaml --stage all
```

The `all` stage passes the newly trained adapter directly to feature extraction,
so changing the training output directory does not disconnect the pipeline.

The lightweight method checks (joint-loss decomposition, final-token pooling,
fusion order, and fold-local preprocessing layout) run on CPU:

```bash
python -m unittest discover -s tests -v
```

Relative paths in the YAML are resolved from the repository root, so no source
file contains a private absolute experiment path.

## Reproducibility boundary

The repository does not include:

- paper-final LoRA checkpoints or projection-head weights;
- dataset-specific final/best parameters or Optuna databases;
- fixed train/validation/test sample indices;
- extracted `h_A/h_B/h_C` or fused features;
- fitted StandardScaler/PCA/SMOTE objects;
- a trained stacking model;
- datasets that cannot legally be redistributed.

All configured output files are generated locally when a user runs the code and
must not be committed as paper assets. `save_local_search_artifacts` is disabled
by default. Example settings and search ranges are provided solely to demonstrate
the executable method; they are not the paper's final per-dataset configuration.
Consequently, exact reproduction of Table 2 is not claimed.

## Method-to-code map

| Paper component | Code |
| --- | --- |
| Dataset-configured three views | `prompts/common.py`, `definition_view.py`, `behavior_view.py`, `holistic_view.py` |
| Shared 4-bit Llama + LoRA + CE | `src/train_lora.py` |
| Per-view projection/SupCon/Triplet | `src/losses.py`, `src/train_lora.py` |
| Raw final-position hidden states | `src/extract_features.py` |
| Difference/product/mean fusion | `src/feature_fusion.py` |
| Fold-local preprocessing and stacking | `src/train_ensemble.py` |
| Accuracy and Weighted-F1 | `src/evaluate.py` |
