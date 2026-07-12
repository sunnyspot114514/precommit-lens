# v4d Final Qwen3.5-4B Report

v4d is complete. Stage 1 passed the frozen benchmark-feasibility gate, but the confirmatory residual-accessibility gate **FAILS** with no winning checkpoint.

## Stage 1: Deployment Transfer

| deployment | eligible | mixed | exact candidate outputs | A/B-switch prompts |
|---|---:|---:|---:|---:|
| Ollama Q4_K_M | 34/64 | 56/64 | 962/1024 | 52/64 |
| Transformers FP16 | 36/64 | 55/64 | 952/1024 | 53/64 |

The FP16 run passes every pre-registered trigger and freezes 33 prompts from 22 template families into a 17/8/8 train/validation/test split. Benchmark viability therefore survives the backend and precision change. This does not identify which model or deployment component produces the stochasticity.

## Confirmatory Result

All 17 train, 8 validation, and 8 test prompts remain mixed under 32 fresh seeds, so the result is conclusive rather than a yield failure.

| checkpoint | prompts | risks | residual L21 | TF-IDF | next-token | judge | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 3 | 0.500 | 0.500 | 0.500 | 0.500 | +0.000 |
| 4 | 8 | 3 | 0.500 | 0.500 | 0.500 | 0.500 | +0.000 |
| 6 | 5 | 2 | 0.500 | 0.500 | 0.500 | 0.500 | +0.000 |
| 8 | 3 | 1 | 0.500 | 0.500 | 0.500 | 0.500 | +0.000 |
| 10 | 3 | 1 | 0.500 | 0.500 | 0.500 | 0.500 | +0.000 |
| 12 | 3 | 1 | 0.500 | 0.500 | 0.500 | 0.500 | +0.000 |

Every primary method is exactly at chance. No checkpoint approaches the frozen `+0.03` residual margin, and no secondary captured layer changes that result.

## Why the Curve Is Exactly 0.5

The post-hoc identity diagnostic changes no gate and samples no new checkpoint. At checkpoints 0-16, every evaluable prompt has one visible prefix and exactly zero within-prompt residual span at all six captured layers. Across test pairs, 1,014 early-spoiler/fake-commit pairs first diverge exactly at policy landing. The 714 hidden-fields pairs diverge two tokens before landing, but that window falls between frozen checkpoints 16 and 24; at checkpoint 24 only two prompts from one risk remain evaluable.

Thus v4d finds abundant outcome stochasticity without an operationally usable signal at the frozen early checkpoints. It does not rule out a narrow two-token hidden-fields signal that this protocol was not powered to evaluate.

## Cost and Stop

Six-layer capture costs `1.005x` plain generation (95% paired CI `0.953`-`1.062`, 16 pairs). The same-model judge costs `231.2` ms per unique prefix and is also at chance.

Per the frozen stopping boundary, v4d ends experimental development. No v4e or pre-submission scale curve is derived from this result.
