# Regex introspection results

Final checkpoint per run and level; means over characters; 95% paired bootstrap CIs.
Faithfulness = agreement of the reported regex with the model's own accept/reject decisions on held-out strings (chance about 0.5).

| run | level | step | decision acc | accept rate | faithfulness | balanced | shuffled | base (step 0) | faith - shuffled | faith - base | error-string faith - 0.5 | exact match | observer | faith - observer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b-s2_L4 | 4 | 4000 | 1.000 | 0.50 | 0.615 | 0.615 | 0.494 | 0.228 | +0.122 [0.111, 0.133] | +0.387 [0.365, 0.409] | +nan [nan, nan] (n=0) | 0.001 | 0.515 | +0.101 [0.088, 0.114] |
