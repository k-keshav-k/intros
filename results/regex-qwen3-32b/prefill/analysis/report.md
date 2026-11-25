# Regex introspection results

Final checkpoint per run and level; means over characters; 95% paired bootstrap CIs.
Faithfulness = agreement of the reported regex with the model's own accept/reject decisions on held-out strings (chance about 0.5).

| run | level | step | decision acc | accept rate | faithfulness | balanced | shuffled | base (step 0) | faith - shuffled | faith - base | error-string faith - 0.5 | exact match | observer | faith - observer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b-prefill_L1 | 1 | 4000 | 1.000 | 0.50 | 0.677 | 0.677 | 0.483 | 0.222 | +0.194 [0.163, 0.225] | +0.454 [0.420, 0.487] | +nan [nan, nan] (n=0) | 0.282 | 0.647 | +0.030 [-0.004, 0.063] |
| regex-qwen3-32b-prefill_L2 | 2 | 4000 | 0.995 | 0.49 | 0.613 | 0.612 | 0.485 | 0.232 | +0.129 [0.108, 0.150] | +0.382 [0.353, 0.410] | -0.360 [-0.439, -0.257] (n=28) | 0.038 | 0.636 | -0.023 [-0.047, 0.001] |
| regex-qwen3-32b-prefill_L3 | 3 | 4000 | 0.996 | 0.50 | 0.604 | 0.602 | 0.494 | 0.245 | +0.110 [0.097, 0.124] | +0.358 [0.334, 0.382] | +0.215 [0.088, 0.348] (n=8) | 0.007 | 0.630 | -0.027 [-0.047, -0.008] |
| regex-qwen3-32b-prefill_L4 | 4 | 4000 | 0.998 | 0.50 | 0.566 | 0.565 | 0.495 | 0.233 | +0.071 [0.059, 0.083] | +0.333 [0.308, 0.356] | +0.230 [0.080, 0.351] (n=6) | 0.000 | 0.518 | +0.048 [0.036, 0.060] |
