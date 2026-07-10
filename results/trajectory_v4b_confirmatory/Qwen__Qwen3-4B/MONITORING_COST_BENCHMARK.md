# v4 Monitoring Cost Benchmark

- frozen test prompts: `9`
- paired runs: `18`
- outputs identical across modes: `True`
- plain generation seconds: `21.660`
- capture seconds: `22.223`
- capture/plain ratio: `1.026` (95% paired bootstrap CI `1.016`-`1.037`)
- mean paired overhead: `0.031` seconds/trajectory
- CUDA peak allocated / reserved GiB: `7.644` / `7.697`

This measures the current 6-layer, 9-checkpoint research capture path. It is not an estimate for a production hook that exports only layer 23.
