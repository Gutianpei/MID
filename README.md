# MID
Code for CVPR 2022 paper "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion"

By Tianpei Gu*, Guangyi Chen*, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou and Jiwen Lu

[[Paper]](https://arxiv.org/abs/2203.13777) |  [[Video Presentation]](https://www.youtube.com/watch?v=g1vf9wio6VM)

[MID GIF]


# Code

## Environment
    PyTorch == 1.7.1
    CUDA > 10.1

## Prepare Data

We pre-process the data of both ETH/UCY dataset and Stanford Drone dataset for training.

To do so run

```
python process_data.py
```

Please see ```process_data.py``` for detail.

## Training

### Step 1: Modify or create your own config file in ```/configs``` 

You can adjust parameters in config file as you like and change the network architecture of the diffusion model in ```models/diffusion.py```

Make sure the ```eval_mode``` is set to False
 
 ### Step 2: Train MID
 
 ```python main.py --config configs/YOUR_CONFIG.yaml --dataset DATASET``` 
 
 Note that ```$DATASET``` should from ["eth", "hotel", "univ", "zara1", "zara2", "sdd"]
 
Logs and checkpoints will be automatically saved.

## Evaluation

To evaluate a trained-model, please set ```eval_mode``` in config file to True and set the epoch you'd like to evaluate at from ```eval_at```

Since diffusion model is an iterative process, the evaluation process may take a long time. We are working on a faster version of MID or you can set a shorter diffusion steps (default 100 steps).

## Result



### Citation

    @inproceedings{gu2022stochastic,
      title={Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion},
      author={Gu, Tianpei and Chen, Guangyi and Li, Junlong and Lin, Chunze and Rao, Yongming and Zhou, Jie and Lu, Jiwen},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={17113--17122},
      year={2022}
    }

### License

Our code is released under MIT License.
