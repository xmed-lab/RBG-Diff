# Checkpoints

Pretrained weights are hosted on Hugging Face (not bundled in Git due to size):
https://huggingface.co/HajihajihaJimmy/RBG-Diff

```shell
HF_HUB_ENABLE_HF_TRANSFER=1 hf download HajihajihaJimmy/RBG-Diff --local-dir ./checkpoints
```

This places `rbgdiff_sim_ema19.pkl` and `rbgdiff_real_ema19.pkl` here; `test.sh` and
`eval_real.py` pick them up by default.
