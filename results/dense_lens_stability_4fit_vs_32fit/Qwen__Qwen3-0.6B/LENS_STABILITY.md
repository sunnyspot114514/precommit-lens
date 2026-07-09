# Dense J-Lens Stability Comparison

- model: `Qwen/Qwen3-0.6B`
- lens A: `4fit`
- lens B: `32fit`
- cases: `200`
- condition filter: `concept_present_target_absent`
- layers: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`

## Aggregate

| metric | mean | median | min | max |
|---|---:|---:|---:|---:|
| spearman | 0.895 | 0.899 | 0.866 | 0.930 |
| pearson | 0.937 | 0.941 | 0.914 | 0.957 |

## By Risk

| risk | cases | spearman mean | spearman median | top-1 agreement |
|---|---:|---:|---:|---:|
| early_spoiler | 50 | 0.920 | 0.918 | 0.400 |
| fake_commit | 50 | 0.876 | 0.875 | 0.400 |
| hidden_fields | 50 | 0.907 | 0.907 | 1.000 |
| schema_bypass | 50 | 0.876 | 0.877 | 1.000 |

## Interpretation

- High rank correlation means the fitted dense lens is stable to fitting-prompt choice.
- Low top-1 agreement means individual headline tokens are unstable even if broad ranking is correlated.
- This ablation should be run after fitting the 32-prompt lens artifact.
