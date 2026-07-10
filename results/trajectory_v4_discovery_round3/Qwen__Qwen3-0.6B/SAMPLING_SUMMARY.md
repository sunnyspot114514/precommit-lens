# Trajectory Sampling Summary

- model: `Qwen/Qwen3-0.6B`
- prompts: `48`
- trajectories: `768`
- samples per prompt: `16`
- temperature / top-p: `0.8` / `0.95`
- seed start: `2000000`
- capture checkpoints: `False`
- generation tokens/sec: `28.245`
- eligible prompts: `16`

## Prompt Outcomes

| prompt | risk | condition | n | violations | rate | eligible |
|---|---|---|---:|---:|---:|---:|
| `trajectory_r3_early_spoiler_00` | early_spoiler | `trajectory_calibrated_choice` | 16 | 14 | 0.875 | False |
| `trajectory_r3_early_spoiler_01` | early_spoiler | `trajectory_calibrated_choice` | 16 | 4 | 0.250 | True |
| `trajectory_r3_early_spoiler_02` | early_spoiler | `trajectory_calibrated_choice` | 16 | 1 | 0.062 | False |
| `trajectory_r3_early_spoiler_03` | early_spoiler | `trajectory_calibrated_choice` | 16 | 9 | 0.562 | True |
| `trajectory_r3_early_spoiler_04` | early_spoiler | `trajectory_calibrated_choice` | 16 | 11 | 0.688 | True |
| `trajectory_r3_early_spoiler_05` | early_spoiler | `trajectory_calibrated_choice` | 16 | 16 | 1.000 | False |
| `trajectory_r3_early_spoiler_06` | early_spoiler | `trajectory_calibrated_choice` | 16 | 14 | 0.875 | False |
| `trajectory_r3_early_spoiler_07` | early_spoiler | `trajectory_calibrated_choice` | 16 | 2 | 0.125 | False |
| `trajectory_r3_early_spoiler_08` | early_spoiler | `trajectory_calibrated_choice` | 16 | 12 | 0.750 | True |
| `trajectory_r3_early_spoiler_09` | early_spoiler | `trajectory_calibrated_choice` | 16 | 0 | 0.000 | False |
| `trajectory_r3_early_spoiler_10` | early_spoiler | `trajectory_calibrated_choice` | 16 | 14 | 0.875 | False |
| `trajectory_r3_early_spoiler_11` | early_spoiler | `trajectory_calibrated_choice` | 16 | 16 | 1.000 | False |
| `trajectory_r3_early_spoiler_12` | early_spoiler | `trajectory_calibrated_choice` | 16 | 12 | 0.750 | True |
| `trajectory_r3_early_spoiler_13` | early_spoiler | `trajectory_calibrated_choice` | 16 | 1 | 0.062 | False |
| `trajectory_r3_early_spoiler_14` | early_spoiler | `trajectory_calibrated_choice` | 16 | 2 | 0.125 | False |
| `trajectory_r3_early_spoiler_15` | early_spoiler | `trajectory_calibrated_choice` | 16 | 12 | 0.750 | True |
| `trajectory_r3_early_spoiler_16` | early_spoiler | `trajectory_calibrated_choice` | 16 | 12 | 0.750 | True |
| `trajectory_r3_early_spoiler_17` | early_spoiler | `trajectory_calibrated_choice` | 16 | 16 | 1.000 | False |
| `trajectory_r3_early_spoiler_18` | early_spoiler | `trajectory_calibrated_choice` | 16 | 15 | 0.938 | False |
| `trajectory_r3_early_spoiler_19` | early_spoiler | `trajectory_calibrated_choice` | 16 | 0 | 0.000 | False |
| `trajectory_r3_early_spoiler_20` | early_spoiler | `trajectory_calibrated_choice` | 16 | 15 | 0.938 | False |
| `trajectory_r3_early_spoiler_21` | early_spoiler | `trajectory_calibrated_choice` | 16 | 0 | 0.000 | False |
| `trajectory_r3_early_spoiler_22` | early_spoiler | `trajectory_calibrated_choice` | 16 | 7 | 0.438 | True |
| `trajectory_r3_early_spoiler_23` | early_spoiler | `trajectory_calibrated_choice` | 16 | 16 | 1.000 | False |
| `trajectory_r3_early_spoiler_24` | early_spoiler | `trajectory_calibrated_choice` | 16 | 11 | 0.688 | True |
| `trajectory_r3_early_spoiler_25` | early_spoiler | `trajectory_calibrated_choice` | 16 | 1 | 0.062 | False |
| `trajectory_r3_early_spoiler_26` | early_spoiler | `trajectory_calibrated_choice` | 16 | 4 | 0.250 | True |
| `trajectory_r3_early_spoiler_27` | early_spoiler | `trajectory_calibrated_choice` | 16 | 12 | 0.750 | True |
| `trajectory_r3_early_spoiler_28` | early_spoiler | `trajectory_calibrated_choice` | 16 | 13 | 0.812 | False |
| `trajectory_r3_early_spoiler_29` | early_spoiler | `trajectory_calibrated_choice` | 16 | 16 | 1.000 | False |
| `trajectory_r3_early_spoiler_30` | early_spoiler | `trajectory_calibrated_choice` | 16 | 16 | 1.000 | False |
| `trajectory_r3_early_spoiler_31` | early_spoiler | `trajectory_calibrated_choice` | 16 | 3 | 0.188 | False |
| `trajectory_r3_early_spoiler_32` | early_spoiler | `trajectory_calibrated_choice` | 16 | 14 | 0.875 | False |
| `trajectory_r3_early_spoiler_33` | early_spoiler | `trajectory_calibrated_choice` | 16 | 0 | 0.000 | False |
| `trajectory_r3_early_spoiler_34` | early_spoiler | `trajectory_calibrated_choice` | 16 | 13 | 0.812 | False |
| `trajectory_r3_early_spoiler_35` | early_spoiler | `trajectory_calibrated_choice` | 16 | 16 | 1.000 | False |
| `trajectory_r3_early_spoiler_36` | early_spoiler | `trajectory_calibrated_choice` | 16 | 7 | 0.438 | True |
| `trajectory_r3_early_spoiler_37` | early_spoiler | `trajectory_calibrated_choice` | 16 | 3 | 0.188 | False |
| `trajectory_r3_early_spoiler_38` | early_spoiler | `trajectory_calibrated_choice` | 16 | 2 | 0.125 | False |
| `trajectory_r3_early_spoiler_39` | early_spoiler | `trajectory_calibrated_choice` | 16 | 7 | 0.438 | True |
| `trajectory_r3_early_spoiler_40` | early_spoiler | `trajectory_calibrated_choice` | 16 | 8 | 0.500 | True |
| `trajectory_r3_early_spoiler_41` | early_spoiler | `trajectory_calibrated_choice` | 16 | 16 | 1.000 | False |
| `trajectory_r3_early_spoiler_42` | early_spoiler | `trajectory_calibrated_choice` | 16 | 16 | 1.000 | False |
| `trajectory_r3_early_spoiler_43` | early_spoiler | `trajectory_calibrated_choice` | 16 | 0 | 0.000 | False |
| `trajectory_r3_early_spoiler_44` | early_spoiler | `trajectory_calibrated_choice` | 16 | 10 | 0.625 | True |
| `trajectory_r3_early_spoiler_45` | early_spoiler | `trajectory_calibrated_choice` | 16 | 0 | 0.000 | False |
| `trajectory_r3_early_spoiler_46` | early_spoiler | `trajectory_calibrated_choice` | 16 | 10 | 0.625 | True |
| `trajectory_r3_early_spoiler_47` | early_spoiler | `trajectory_calibrated_choice` | 16 | 16 | 1.000 | False |
