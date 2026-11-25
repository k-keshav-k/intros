"""From-scratch small GPT-2 version of the regex experiment (port of boolintro.tiny).

1. Pretrain a randomly initialised GPT-2 on single-string accept/reject decisions for one
   group of characters, plus their hidden regex ("pre_report"), and decisions only for a
   control group ("pre_choice").
2. Finetune on decisions only for a new group ("new"), whose character tokens get fresh
   random embeddings, and test whether the model can write their regexes.
"""
