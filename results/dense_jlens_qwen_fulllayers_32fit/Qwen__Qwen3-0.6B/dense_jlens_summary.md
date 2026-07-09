# Dense J-Lens Probe

- model: `Qwen/Qwen3-0.6B`
- layers: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`
- fit prompts: `32`
- dtype: `float16`
- device: `cuda`

## Case Summary

| case | group | validator | prompt term hits | best J-lens watched token | output preview |
|---|---|---|---|---|---|

## Interpretation Notes

- This run fits actual dense local Jacobian matrices for the selected layers.
- The default smoke run is not yet a full-depth corpus-averaged lens.
- Prompt term hits flag cases where a watched word appears directly in the prompt.
