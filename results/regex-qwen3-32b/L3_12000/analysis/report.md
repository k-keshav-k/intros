# Regex introspection results

Final checkpoint per run and level; means over characters; 95% paired bootstrap CIs.
Faithfulness = agreement of the reported regex with the model's own accept/reject decisions on held-out strings (chance about 0.5).

| run | level | step | decision acc | accept rate | faithfulness | balanced | shuffled | base (step 0) | faith - shuffled | faith - base | error-string faith - 0.5 | exact match | observer | faith - observer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L3 | 3 | 12000 | 0.996 | 0.50 | 0.604 | 0.601 | 0.494 | 0.282 | +0.110 [0.084, 0.140] | +0.327 [0.291, 0.360] | -0.088 [-0.294, 0.118] (n=17) | 0.029 | 0.610 | -0.005 [-0.034, 0.027] |
