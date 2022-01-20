# Homomorphic Policy Gradient Algorithms
* This supplementary material includes our implementation of DHPG and all 
the baseline algorithms used in the paper: 
  * **Pixel observations:** DHPG, DBC, DeepMDP, SAC-AE, DrQ-v2.
  * **State observations:** DHPG, TD3, DDPG, SAC.
* Results were obtained on Python v3.8.10, CUDA v11.4, PyTorch v1.10.0 on 10 seeds.
* Our code will be publicly released after the review process. 

## Setup
* Install the following libraries:
```commandline
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```
* Install [Mujoco](https://github.com/deepmind/mujoco) and [DeepMind Control Suite](https://github.com/deepmind/dm_control)
following the official instructions.
* We recommend using a conda virtual environment to run the code.
Create the virtual environment:
```commandline
conda create -n hpg_env python=3.8
conda activate hpg_env
pip install --upgrade pip
```
* Install dependencies:
```commandline
pip install -r requirements.txt
````

## Instructions
### Training on Pixel Observations
* To train agents on pixel observations:
```commandline
python train.py task=pendulum_swingup agent=hpg 
```
* Available agents are `hpg`, `hpg_aug`, `hpg_ind`, `hpg_ind_aug`, 
`dbc`, `deepmdp`, `sacae`, `drqv2`.
  * `hpg` is the DHPG variant in which gradients of HPG and DPG are summed 
  together for a single actor update (`hpg_aug` is `hpg` with image augmentation.) 
  * `hpg_ind` is the DHPG variant in which gradients of HPG and DPG are 
   used to independently update the actor (`hpg_ind_aug` is `hpg_ind` with image augmentation.)   
* If you do not have a CUDA device, use `device=cpu`.

### Training on State Observations
* To train agents on pixel observations:
```commandline
python train.py pixel_obs=false action_repeat=1 frame_stack=1 task=pendulum_swingup agent=hpg 
```
* Available agents are `hpg`, `hpg_ind`, `td3`, `sac`, `ddpg_original`, `ddpg_ours`.

### Tensorboard
* To monitor results use:
```commandline
tensorboard --logdir exp
```