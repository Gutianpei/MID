# Official Implementation for CVPR2022 Paper "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion"

By Tianpei Gu*, Guangyi Chen*, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou and Jiwen Lu

## Environment

We use pytorch 1.7.1 with cuda > 10.1 for all experiments.


## Prepare Data

```
python process_data.py
```

## Train & Test

First modify or create your own config file in ```/configs``` and run ```python main.py --config configs/YOUR_CONFIG.yaml --dataset DATASET``` where ```$DATASET``` should from ["eth", "hotel", "univ", "zara1", "zara2", "sdd"]
