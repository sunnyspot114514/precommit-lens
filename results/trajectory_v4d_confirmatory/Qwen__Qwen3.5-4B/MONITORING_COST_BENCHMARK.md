# v4 Monitoring Cost Benchmark

- frozen test prompts: `8`
- paired runs: `16`
- outputs identical across modes: `True`
- plain generation seconds: `21.645`
- capture seconds: `21.762`
- capture/plain ratio: `1.005` (95% paired bootstrap CI `0.953`-`1.062`)
- mean paired overhead: `0.007` seconds/trajectory
- CUDA peak allocated / reserved GiB: `8.042` / `8.102`

This measures the current 6-layer, 9-checkpoint research capture path. It is not an estimate for a production hook that exports only layer 21.
