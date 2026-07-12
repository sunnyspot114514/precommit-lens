# v4d Stage-One FP16 Transfer Gate

- status: **PASS**
- stage 2 triggered: `True`
- raw eligible prompts: `36/64`
- supported risks: `['early_spoiler', 'fake_commit', 'hidden_fields']`
- selected prompts / families: `33` / `22`
- grouped split: `{'train': 17, 'validation': 8, 'test': 8}`
- outcome states C / R / mixed: `8 / 1 / 55`
- exact candidate outputs: `952/1024` (`93.0%`)
- prompts switching between exact A/B: `53/64`
- generation throughput: `16.443` tok/s

## Frozen Checks

| check | pass |
|---|---:|
| `raw_eligible_prompts` | `True` |
| `selected_prompts` | `True` |
| `supported_risks` | `True` |
| `selected_template_families` | `True` |
| `grouped_split_support` | `True` |

## Risk Support

| risk | eligible prompts | eligible families |
|---|---:|---:|
| `early_spoiler` | 10 | 7 |
| `fake_commit` | 12 | 8 |
| `hidden_fields` | 11 | 7 |
| `schema_bypass` | 3 | 2 |

A pass authorizes only the frozen v4d confirmatory stage. A stop identifies deployment-state dependence but does not isolate model generation, quantization, backend, or chat rendering.
