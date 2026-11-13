# Evaluation protocol and released conditions

This document describes the direct-model protocol of SILICA and every model
condition whose responses are released with the dataset (`predictions/` on
the Hugging Face Hub), with the settings recorded in
`predictions/conditions.json`.

## The released prompt

Every row is rendered by `silica prompts` (`silica.prompts.render_prompt`) as
two chat messages. The system message is the same for every row:

> You are solving the SILICA-Hard posterior-matched paired-query identifiability reasoning benchmark. The supplied declaration defines an exact finite hypothesis class: enumerate all Boolean feature assignments, retain only assignments accepted by every declared Boolean-program constraint, and then retain only assignments consistent with every observed form/meaning pair. Evaluate the query under every surviving assignment. Return ANSWER only if all survivors produce one identical canonical answer; otherwise return ABSTAIN. Reason silently. Emit exactly one JSON object satisfying canonical_response_schema, with no markdown, rationale, confidence, or additional keys.

The user message is the canonical JSON (sorted keys, no whitespace) of:

| Key | Content |
|---|---|
| `prompt_version`, `adapter_task`, `adapter_version`, `adapter_protocol_hash`, `response_schema_version` | protocol metadata |
| `response_task` | `lexicon` or `grammar_feature` |
| `hypothesis_declaration` | 14 declaration fields: `type`, `schema_family`, `base_class_size`, `class_size`, `boolean_dsl_semantics`, `boolean_program`, `feature_variables_lsb_to_msb`, `exact_enumeration`, `lexical_axes`, `fixed_lexicon`, `grammar_axes`, `grammar_semantics`, `constant_markers`, `constant_rules` |
| `observations` | the three observations as `{channel, category, semantic, form}` |
| `query` | the query |
| `decision_rule` | when to `ANSWER` and when to `ABSTAIN` |
| `canonical_response_schema` | the JSON schema of a valid response, including the answer domain |

No worked example is given. The model never sees family, template, route or
grouping ids, observation ids, semantic or class hashes, structural
statistics, survivors, witnesses or any gold field. The 12,288 rendered
prompts have the prompt-equivalence identity
`cf36ed363351ffed8990c9f2ff14f0955b6ad3eca08afbcdca6b7c42a62efb98`
(`prompt_ledger_sha256` of `silica prompts --cohort full`).

## Scoring

Only the final content (`response_text`) is scored; a reasoning trace
(`reasoning_text`) never is. A row is valid only if the request succeeded,
`finish_reason` is `stop`, and the final content is exactly one JSON object
`{"decision": "ANSWER", "answer": <string in the answer domain>}` or
`{"decision": "ABSTAIN", "answer": null}`. Nothing is repaired, and invalid
rows count as wrong. The common cohort is 1,536 rows: one complete eight-row
route group for each of the 192 templates (cohort hash
`7956f0fb7270c983bb27a76ace8f2130c370bd9db64e37cbd191bfe7ead51fdc`).

## Released conditions

| Condition | Model | Engine | dtype | Decoding (T / top-p / top-k / max tokens / seed) | Thinking | Prompt | Rows | Status |
|---|---|---|---|---|---|---|---:|---|
| `qwen25-05b` | Qwen/Qwen2.5-0.5B-Instruct | Transformers 4.44.2 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | released | 1,536 | complete |
| `qwen25-15b` | Qwen/Qwen2.5-1.5B-Instruct | Transformers 4.44.2 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | released | 1,536 | complete |
| `smollm2-17b` | HuggingFaceTB/SmolLM2-1.7B-Instruct | Transformers 4.44.2 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | released | 1,536 | complete |
| `qwen25-3b` | Qwen/Qwen2.5-3B-Instruct | Transformers 4.44.2 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | released | 1,536 | complete |
| `phi35-mini` | microsoft/Phi-3.5-mini-instruct | Transformers 4.44.2 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | released | 1,536 | complete |
| `qwen25-7b` | Qwen/Qwen2.5-7B-Instruct | Transformers 4.44.2 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | released | 1,536 | complete |
| `qwen25-14b` | Qwen/Qwen2.5-14B-Instruct | Transformers 4.44.2 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | released | 1,536 | complete |
| `qwen25-32b` | Qwen/Qwen2.5-32B-Instruct | Transformers 4.44.2 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | released | 1,536 | complete |
| `qwen25-14b-full` | Qwen/Qwen2.5-14B-Instruct | Transformers 4.44.2 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | released | 12,288 | complete |
| `qwen25-15b-replicate` | Qwen/Qwen2.5-1.5B-Instruct | Transformers 4.44.2 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | released | 1,536 | complete |
| `qwen3-235b-trace-on` | Qwen/Qwen3-235B-A22B-GPTQ-Int4 | vLLM 0.18.1 | float16 | 0.6 / 0.95 / 20 / 16,384 / 0 | on | released | 1,536 | complete |
| `qwen3-235b-trace-off` | Qwen/Qwen3-235B-A22B-GPTQ-Int4 | vLLM 0.18.1 | float16 | 0.6 / 0.95 / 20 / 16,384 / 0 | off | released | 1,536 | complete |
| `qwen3-17b-trace-off` | Qwen/Qwen3-1.7B | vLLM 0.18.1 | bfloat16 | 1.0 / 1.0 / 20 / 16,384 / 0 | off | released | 1,536 | complete |
| `qwen3-17b-trace-on` | Qwen/Qwen3-1.7B | vLLM 0.18.1 | bfloat16 | 1.0 / 1.0 / 20 / 16,384 / 0 | on | released | 1,536 | complete |
| `qwen35-2b-greedy` | Qwen/Qwen3.5-2B | vLLM 0.18.1 | bfloat16 | 0 / 1.0 / -1 / 16,384 / 0 | off | released | 1,536 | complete |
| `qwen35-2b-stochastic` | Qwen/Qwen3.5-2B | vLLM 0.18.1 | bfloat16 | 1.0 / 1.0 / 20 / 16,384 / 0 | off | released | 1,536 | complete |
| `qwen35-2b-trace-on` | Qwen/Qwen3.5-2B | vLLM 0.18.1 | bfloat16 | 1.0 / 1.0 / 20 / 16,384 / 0 | on | released | 1,536 | complete except one harness adapter error row |
| `qwen35-08b-trace-on` | Qwen/Qwen3.5-0.8B | vLLM 0.18.1 | bfloat16 | 1.0 / 1.0 / 20 / 16,384 / 0 | on | released | 1,536 | complete except one harness adapter error row |
| `phi35-mini-multitoken-stop` | microsoft/Phi-3.5-mini-instruct | vLLM 0.18.1 | bfloat16 | 0 / 1.0 / -1 / 16,384 / 0 | n/a | released | 314 | partial: 314 of 1,536 rows within the time limit |
| `qwen25-14b-ablation-verbatim` | Qwen/Qwen2.5-14B-Instruct | vLLM 0.18.1 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | released | 1,536 | complete |
| `qwen25-14b-ablation-scratchpad` | Qwen/Qwen2.5-14B-Instruct | vLLM 0.18.1 | bfloat16 | 0 / 1.0 / - / 4,096 / 0 | n/a | scratchpad | 1,536 | complete |
| `qwen25-14b-ablation-lean` | Qwen/Qwen2.5-14B-Instruct | vLLM 0.18.1 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | lean | 1,536 | complete |
| `qwen25-14b-ablation-compact_rule` | Qwen/Qwen2.5-14B-Instruct | vLLM 0.18.1 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | compact_rule | 1,536 | complete |
| `qwen25-32b-ablation-verbatim` | Qwen/Qwen2.5-32B-Instruct | vLLM 0.18.1 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | released | 1,536 | complete |
| `qwen25-32b-ablation-scratchpad` | Qwen/Qwen2.5-32B-Instruct | vLLM 0.18.1 | bfloat16 | 0 / 1.0 / - / 4,096 / 0 | n/a | scratchpad | 1,536 | complete |
| `qwen25-32b-ablation-lean` | Qwen/Qwen2.5-32B-Instruct | vLLM 0.18.1 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | lean | 1,536 | complete |
| `qwen25-32b-ablation-compact_rule` | Qwen/Qwen2.5-32B-Instruct | vLLM 0.18.1 | bfloat16 | 0 / 1.0 / - / 384 / 0 | n/a | compact_rule | 1,536 | complete |

"-" means not set (no top-k truncation). Temperature 0 is greedy decoding.
Model revisions (immutable Hugging Face commits):

| Model | Revision |
|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Qwen/Qwen2.5-3B-Instruct | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Qwen/Qwen2.5-14B-Instruct | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` |
| Qwen/Qwen2.5-32B-Instruct | `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd` |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | `31b70e2e869a7173562077fd711b654946d38674` |
| microsoft/Phi-3.5-mini-instruct | `2fe192450127e6a83f7441aef6e3ca586c338b77` |
| Qwen/Qwen3-235B-A22B-GPTQ-Int4 | `dff138951f3898772fc0d795a4b5b30f417577f4` |
| Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Qwen/Qwen3.5-2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` |

### Sealed panel (Transformers)

The eight common-cohort panel conditions, the full-artifact Qwen2.5-14B run
and the cross-machine Qwen2.5-1.5B replicate ran Hugging Face Transformers on
one GPU in bfloat16 with batch size two, left padding, the tokenizer chat
template, greedy decoding and at most 384 new tokens. Generation had to end
with the end-of-sequence token: `finish_reason` is `stop` when it did and
`length` when it reached the cap. `response_text` is the decoded content
before that token, with any other special token kept in the text, so a
special token inside the content makes the row invalid.

### Trace-permitted and matched-control conditions (vLLM)

These ran vLLM 0.18.1 through the tokenizer chat template. Thinking is the
template's `enable_thinking` switch; with thinking on, the `deepseek_r1`
reasoning parser separates the trace (`reasoning_text`) from the final
content (`response_text`), and only the final content is scored. Within each
matched pair the checkpoint, revision, prompt, sampling settings, token budget,
seed and engine version are identical:

- Qwen3-235B-A22B and Qwen3-1.7B: trace on versus off (thinking switch and
  reasoning parser only). Both Qwen3-235B arms load the same GPTQ Int4
  checkpoint and run vLLM's `gptq_marlin` kernel, with tensor parallelism 4.
- Qwen3.5-2B: greedy versus stochastic decoding, thinking off (temperature and
  top-k only).

The Qwen3.5-2B and Qwen3.5-0.8B trace-on conditions each lost one row to an
adapter error in the evaluation harness. That row is released with
`request_error: true` and counts as invalid; the paper reports only these
conditions' trace termination (`finish_reason` `length`) and strict format
validity.

### Phi-3.5 with a multi-token stop

Under the panel's single end-of-sequence contract every Phi-3.5-mini response
carries its end-of-turn token inside the scored content. The replacement
condition `phi35-mini-multitoken-stop` keeps the checkpoint, revision, cohort,
prompt and scoring and serves the model through vLLM, which stops on the
checkpoint's declared end-of-turn tokens, with greedy decoding and a
16,384-token budget. It reached 314 of the 1,536 common-cohort rows within its
time limit; `silica evaluate` requires a complete cohort, so
`scripts/reproduce_paper.py` scores this file row by row.

## Prompt ablations

Four renderings of the same 1,536-row cohort were served to Qwen2.5-14B and,
separately, Qwen2.5-32B through vLLM 0.18.1 (bfloat16, temperature 0, top-p
1.0, seed 0). Each arm changes one thing relative to the released prompt.

### `verbatim`

The released system and user messages, byte for byte (`prompt_variant`
`released`). It checks the ablation harness against the sealed panel rows of
the same checkpoints.

### `scratchpad`

The user message is unchanged; the system message is replaced by:

> You are solving the SILICA-Hard posterior-matched paired-query identifiability reasoning benchmark. The supplied declaration defines an exact finite hypothesis class: enumerate all Boolean feature assignments, retain only assignments accepted by every declared Boolean-program constraint, and then retain only assignments consistent with every observed form/meaning pair. Evaluate the query under every surviving assignment. Return ANSWER only if all survivors produce one identical canonical answer; otherwise return ABSTAIN. Work the derivation out in writing first, inside one <think> ... </think> block, taking as many steps as you need. After </think> emit exactly one JSON object satisfying canonical_response_schema, with no markdown, rationale, confidence, or additional keys.

The budget is 4,096 new tokens. The scored final content is the text after
the last `</think>`, and the text before it is released as `reasoning_text`.
If the block was never closed, the scored content is the span from the first
`{` to the last `}` of the completion (with the whole completion released as
`reasoning_text`), or the whole completion if it contains no such span.

### `lean`

The system message is unchanged, and keys are removed from the user message.
The arm's drop list names four top-level keys and eleven declaration keys.
Nine of them occur in the released user message, so the lean user message
differs from it by exactly these keys:

- top-level: `adapter_protocol_hash` (the only hash the released user message
  carries), `adapter_task`, `adapter_version`, `response_schema_version`;
- `hypothesis_declaration`: `grammar_semantics` (the grammar realization
  block), `constant_markers`, `constant_rules`, `fixed_lexicon`,
  `schema_family`.

The other six keys on the drop list, `base_class_hash`, `class_hash`,
`class_semantic_hash`, `boolean_program_hash`, `class_semantic_hash_definition`
and `posterior_matching_contract`, are fields of the public row that the
released renderer never includes, so removing them changes nothing. Of the
six metadata fields the paper lists (the posterior-matching contract, schema
family, class-hash definition, adapter task, adapter version and response
schema version), the posterior-matching contract and the class-hash definition
are therefore among these no-op removals; the other four are removed. The
lean user message is 25.3% shorter on average, and it keeps `grammar_axes`
and every other field the prompt-only solver reads, which still solves all
12,288 lean prompts.

### `compact_rule`

The user message is unchanged; the system message replaces the enumeration
instruction with the guarded-switch route:

> You are solving the SILICA-Hard posterior-matched paired-query identifiability reasoning benchmark. The declared Boolean program is a guarded switch between two query-facing features. The three dictionary observations fix the value of three feature bits. Substitute those three bits into the program, and decide which of the two query-facing features the program then forces to a single value: a query on that feature is answerable, and a query on the other one is not, because both of its values still occur among the compatible assignments. Return ANSWER with that forced value only if the query asks about the forced feature; otherwise return ABSTAIN. You do not need to enumerate the class. Reason silently. Emit exactly one JSON object satisfying canonical_response_schema, with no markdown, rationale, confidence, or additional keys.

Because it states structure the released prompt leaves to be inferred, it is
an upper bound on what relieving the stated procedure can buy, not a matched
control.

The ablation arms are exploratory: they ran outside the sealed evaluation
stack and are scored by the same strict rule as every other condition.
