# Regex introspection results

Final checkpoint per run and level; means over characters; 95% paired bootstrap CIs.
Faithfulness = agreement of the reported regex with the model's own accept/reject decisions on held-out strings (chance about 0.5).

| run | level | step | decision acc | accept rate | faithfulness | balanced | shuffled | base (step 0) | faith - shuffled | faith - base | error-string faith - 0.5 | exact match | observer | faith - observer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b_L1 | 1 | 4000 | 1.000 | 0.50 | 0.616 | 0.616 | 0.488 | 0.261 | +0.128 [0.101, 0.155] | +0.355 [0.326, 0.385] | +nan [nan, nan] (n=0) | 0.206 | 0.649 | -0.033 [-0.068, 0.002] |
| regex-qwen3-32b_L2 | 2 | 4000 | 0.995 | 0.49 | 0.598 | 0.596 | 0.486 | 0.261 | +0.112 [0.095, 0.130] | +0.337 [0.311, 0.363] | -0.352 [-0.415, -0.275] (n=28) | 0.040 | 0.625 | -0.027 [-0.048, -0.006] |
| regex-qwen3-32b_L3 | 3 | 4000 | 0.996 | 0.50 | 0.586 | 0.584 | 0.495 | 0.282 | +0.090 [0.078, 0.104] | +0.306 [0.283, 0.327] | +0.316 [0.188, 0.458] (n=8) | 0.003 | 0.611 | -0.026 [-0.044, -0.006] |
| regex-qwen3-32b_L4 | 4 | 4000 | 0.998 | 0.50 | 0.552 | 0.552 | 0.497 | 0.268 | +0.055 [0.028, 0.086] | +0.296 [0.259, 0.336] | +0.222 [-0.056, 0.500] (n=2) | 0.000 | 0.515 | +0.029 [0.001, 0.061] |
