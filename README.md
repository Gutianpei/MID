# MID
Code for CVPR 2022 paper "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion"

By Tianpei Gu*, Guangyi Chen*, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou and Jiwen Lu

[[Paper]](https://arxiv.org/abs/2203.13777) |  [[Video Presentation]](https://www.youtube.com/watch?v=g1vf9wio6VM)

<p align="center">
  <img src="https://user-images.githubusercontent.com/21379120/204936740-65891c87-c4c1-467f-a883-8311af89ba09.gif" alt="animated" />
</p>

> Human behavior has the nature of indeterminacy, which requires the pedestrian trajectory prediction system to model the multi-modality of future motion states. Unlike existing stochastic trajectory prediction methods which usually use a latent variable to represent multi-modality, we explicitly simulate the process of human motion variation from indeterminate to determinate. In this paper, we present a new framework to formulate the trajectory prediction task as a reverse process of **motion indeterminacy diffusion (MID)**, in which we progressively discard indeterminacy from all the walkable areas until reaching the desired trajectory. This process is learned with a parameterized Markov chain conditioned by the observed trajectories. We can adjust the length of the chain to control the degree of indeterminacy and balance the diversity and determinacy of the predictions. Specifically, we encode the history behavior information and the social interactions as a state embedding and devise a Transformer-based diffusion model to capture the temporal dependencies of trajectories.

## **Update 04/2023** 


We integrate DDIM into MID framework which only uses **TWO steps** to achieve similar performance, accelerate **50x** speed compares to original 100 steps generation.

The update is a one-line changes in ```models/diffusion.py```. To enable fast sampling, you can change sampling to **ddim** set the step in ```main.py```. Note that the step needs to be factors of your trained diffusion steps (100 in our settings). The fast sampling can directly apply to any **TRAINED** model and does not require re-training with DDIM. 

In our experiment, we are able to achieve 0.41/0.71 for just TWO diffusion steps in ETH dataset with the same trained model in our paper, compares to original 0.39/0.66 with 100 diffusion steps.



# Code

## Environment
    PyTorch == 1.7.1
    CUDA > 10.1

## Prepare Data

The preprocessed data splits for the ETH/UCY and Stanford Dronw datasets are in ```raw_data```. We preprocess the data and generate .pkl files for training.

To do so run

```
python process_data.py
```

The `train/validation/test/` splits are the same as those found in [Social GAN]( https://github.com/agrimgupta92/sgan). Please see ```process_data.py``` for detail.

## Training

### Step 1: Modify or create your own config file in ```/configs``` 

You can adjust parameters in config file as you like and change the network architecture of the diffusion model in ```models/diffusion.py```

Make sure the ```eval_mode``` is set to False
 
 ### Step 2: Train MID
 
 ```python main.py --config configs/YOUR_CONFIG.yaml --dataset DATASET``` 
 
 Note that ```$DATASET``` should from ["eth", "hotel", "univ", "zara1", "zara2", "sdd"]
 
Logs and checkpoints will be automatically saved.

## Evaluation

To evaluate a trained-model, please set ```eval_mode``` in config file to True and set the epoch you'd like to evaluate at from ```eval_at``` and run

 ```python main.py --config configs/YOUR_CONFIG.yaml --dataset DATASET``` 

Since diffusion model is an iterative process, the evaluation process may take a long time. We are working on a faster version of MID or you can set a shorter diffusion steps (default 100 steps).


### Citation
```
    @inproceedings{gu2022stochastic,
      title={Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion},
      author={Gu, Tianpei and Chen, Guangyi and Li, Junlong and Lin, Chunze and Rao, Yongming and Zhou, Jie and Lu, Jiwen},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={17113--17122},
      year={2022}
    }
```
### License

Our code is released under MIT License.
