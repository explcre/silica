# Data format

A SILICA dataset directory (the Hugging Face dataset repository) contains:

```
SHA256SUMS            SHA-256 of the eight shards and both manifests
manifest.json         root manifest
public/manifest.json  public manifest
public/<shard>.jsonl  public rows, one JSON object per line
gold/<shard>.jsonl    gold rows, joined to public rows by id
predictions/          released model responses (see below)
```

There are four shards per view, one per mechanism family
(`hard_posterior_matched__test__<family>.jsonl` for `equality_chain`,
`majority_guard`, `parity_guard` and `selector_multiplexer`), with 3,072 rows
each. Every line is canonical JSON: sorted keys, no insignificant whitespace,
UTF-8. All hashes are SHA-256 of canonical JSON (`silica.util.object_hash`)
unless stated otherwise.

Some identifiers inside the files predate the benchmark's final name and are
kept so that every hash still matches: `dataset_version` is `0.2.0-rc1`,
`track` and the shard names read `hard_posterior_matched`, grouping ids start
with `hard-v2-`, and the model-facing system message names "SILICA-Hard".

## Public rows

| Field | Type | Content |
|---|---|---|
| `id` | string | `silica-` and the first 24 hex digits of the hash of the row without its id |
| `dataset_version` | string | `0.2.0-rc1` |
| `split` | string | `test` |
| `track` | string | `hard_posterior_matched` |
| `task` | string | `lexicon` (the query asks the meaning of a form) or `grammar_feature` (it asks a grammar value) |
| `language_id` | string | id of the micro-language instance |
| `query_id` | string | id of the query |
| `query` | object | `{"category", "form"}` for `lexicon`, `{"feature"}` for `grammar_feature` |
| `evidence` | array | three observations `{"id", "channel", "category", "form", "semantic"}`; `channel` is `dictionary` and `semantic` is the meaning of `form` |
| `hypothesis_declaration` | object | the closed hypothesis class, below |
| `budget` | object | `{"id": "dictionary_exact_3", "dictionary": 3, "parallel": 0}` |
| `policy` | string | `fixed_counterfactual_three_dictionary` |
| `complete_unit_id` | string | the four-cell unit (two `ANSWER` and two `ABSTAIN` rows on one query feature) |
| `paired_query_unit_id` | string | the eight-row route group (two linked four-cell units) |
| `paired_state_id` | string | the same-state query pair (one program and evidence, two queries with opposite gold actions) |
| `counterfactual_cell_id` | string | the cell of the unit |
| `mechanism` | object | `family`, `template_id`, `template_hash`, `route`, and semantic hashes of the class, unit, paired behavior and template |
| `structural_complexity` | object | `operator_count`, `constraint_count`, `maximum_dependency_depth`, `dependency_span`, `variable_support`, `variable_support_size`, `full_evaluation_proof_length` of the program |
| `item_index` | integer | bit position of the queried feature in `feature_variables_lsb_to_msb` (0 to 7) |

`hypothesis_declaration` fields:

| Field | Content |
|---|---|
| `type`, `schema_family` | declaration type and schema |
| `feature_variables_lsb_to_msb` | the eight binary features in bit order, each `{kind, name, values}`: four lexical swaps (`noun_swap_01`, `noun_swap_23`, `verb_swap_01`, `modifier_swap_01`) and four grammar settings (`order_family`, `order_variant`, `alignment`, `agreement`) |
| `boolean_program` | `{dsl_version, variables, constraints}`; each constraint is an expression tree over `var`, `const`, `not`, `and`, `or`, `xor`, `eq`, `implies` and `ite` |
| `boolean_dsl_semantics` | the operator semantics, arities and canonicalization rules, in words |
| `exact_enumeration` | the universe (all 2^8 assignments) and the filter |
| `base_class_size`, `class_size` | 256 and 128 (the assignments the program accepts) |
| `lexical_axes` | per lexical feature: `category`, the two `forms`, the two `possible_meanings`, and the swap constraint |
| `grammar_axes` | per grammar feature: the two canonical values |
| `fixed_lexicon` | forms whose meaning does not vary |
| `grammar_semantics` | how grammar values map to word order, case and agreement marking |
| `constant_markers`, `constant_rules` | fixed morphemes and fixed rules |
| `posterior_matching_contract` | declared class size 128, posterior size 16, three observed features, and the paired-query requirement |
| `base_class_hash`, `class_hash`, `class_semantic_hash`, `class_semantic_hash_definition`, `boolean_program_hash` | content hashes and the definition of the semantic class hash |

The model-facing prompt renders only `type`, `schema_family`,
`base_class_size`, `class_size`, `boolean_dsl_semantics`, `boolean_program`,
`feature_variables_lsb_to_msb`, `exact_enumeration`, `lexical_axes`,
`fixed_lexicon`, `grammar_axes`, `grammar_semantics`, `constant_markers` and
`constant_rules` from the declaration, plus the observations' `channel`,
`category`, `semantic` and `form`, the query, the decision rule and the
response schema (see [protocol.md](protocol.md)). The other public fields are
evaluation metadata for grouping, breakdowns and resampling.

## Gold rows

Gold rows are evaluator-only: a model must never see them.

| Field | Content |
|---|---|
| `id` | the public row id |
| `gold.decision` | `ANSWER` if all 16 languages compatible with the evidence give one answer, else `ABSTAIN` |
| `gold.answer` | the canonical answer for `ANSWER`, null for `ABSTAIN` |
| `gold.target_answer` | the payload the row's four-cell unit is built around (the gold answer on `ANSWER` rows) |
| `certificate` | the proof of the decision: `decision`, `consensus`, `hypothesis_class_size` (128), `hypothesis_count` (16), `support` (each distinct answer with its count and the hash of its hypothesis ids), `disagreement_witness` (two compatible hypotheses with different answers, for `ABSTAIN`), and hashes of the class, program, evidence, query, ordered experiment, answer, support and version space |
| `oracle` | construction metadata: the hidden feature roles (`query_feature`, `role_zero_feature`, `role_one_feature` and their bits and payload values), `observed_feature_bits`, `ordered_evidence_features`, `guard_active`, `minimum_sufficient_proof` (the smallest evidence subset with the same decision and answer), the generated `surface_inventory`, the target hypothesis, the grouping ids, and `public_record_hash` (the hash of the public row) |

`silica verify` recomputes every certificate and gold decision from the
public row alone and checks every manifest and checksum.

## Prediction format

`silica evaluate` reads a JSON Lines file with one object per evaluated row
(`silica.predictions.Prediction`):

| Field | Type | Content |
|---|---|---|
| `id` | string | the public row id |
| `response_text` | string or null | the final content returned by the model; this is what is scored |
| `finish_reason` | `"stop"`, `"length"` or null | `stop` if generation ended on its own, `length` if it reached the token cap |
| `completion_tokens` | integer or null | generated tokens, including any reasoning tokens |
| `request_error` | boolean | true if no response was obtained; `response_text` must then be null |
| `reasoning_text` | string (optional) | a reasoning trace returned separately from the final content; never scored |

No other key is accepted, and ids must be unique. The file must cover either
all 12,288 rows or exactly the 1,536-row common cohort (`silica prompts
--cohort common`). A row
is format-valid only if `request_error` is false, `finish_reason` is `stop`,
and `response_text` is exactly `{"decision": "ANSWER", "answer": <string in
the query's answer domain>}` or `{"decision": "ABSTAIN", "answer": null}`.

## Released model responses

`predictions/<condition_id>.jsonl` holds one condition's responses in the
format above. `predictions/conditions.json` is
`{"description": ..., "conditions": [...]}` with one entry per file:

| Field | Content |
|---|---|
| `condition_id` | file stem |
| `model`, `revision` | Hugging Face repository id and 40-character commit |
| `backend`, `engine_version` | `transformers` or `vllm`, and its version |
| `dtype` | the dtype the weights ran in |
| `quantization` | null, or the quantization actually used |
| `decoding` | `temperature`, `top_p`, `top_k`, `max_new_tokens`, `seed` (null: not set) |
| `thinking_enabled` | true, false, or `not applicable` for models without a thinking mode |
| `reasoning_parser` | the serving engine's reasoning parser, or null |
| `prompt_variant` | `released`, `scratchpad`, `lean` or `compact_rule` |
| `cohort`, `rows` | `common` or `full`, and the number of rows in the file |
| `status` | `complete`, `complete except one harness adapter error row`, or `partial: 314 of 1,536 rows within the time limit` |
| `paper_section` | where the paper reports the condition |
| `predictions_file`, `predictions_sha256` | the file and its SHA-256 |

[protocol.md](protocol.md) describes each condition.
