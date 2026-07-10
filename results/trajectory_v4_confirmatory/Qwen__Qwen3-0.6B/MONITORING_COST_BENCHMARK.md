# v4 Monitoring Cost Benchmark

- frozen test prompts: `9`
- paired runs: `18`
- outputs identical across modes: `True`
- plain generation seconds: `15.331`
- capture seconds: `15.543`
- capture/plain ratio: `1.014` (95% paired bootstrap CI `0.999`-`1.029`)
- mean paired overhead: `0.012` seconds/trajectory

This measures the current six-layer, nine-checkpoint research capture path. It is not an estimate for a production hook that exports only layer 18.
