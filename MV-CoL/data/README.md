# Dataset input format

This directory intentionally contains no paper dataset or fixed split indices.
Supply three independent UTF-8 JSONL files for the selected task:

```text
data/<task>/train.jsonl
data/<task>/validation.jsonl
data/<task>/test.jsonl
```

Each non-empty line must be one JSON object containing at least the configured
text and label fields. An ID is recommended but optional:

```json
{"id": "sample-001", "text": "...", "label": "Constructive"}
```

For datasets whose source labels are numeric, the task YAML supplies an explicit
`label_aliases` mapping. For example, the Question config maps source value `1`
to `Question`. Unknown values raise an error; they are never silently remapped.

The three files must be created independently before running the pipeline. Test
labels are used only during the one-time final evaluation, after hyperparameter
selection and Train+Validation refitting are complete.
