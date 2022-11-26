# Continuous MDP Homomorphisms and Homomorphic Policy Gradients
Author's PyTorch implementation of Deep Homomorphic Policy Gradients (DHPG). 
If you use our code, please cite our NeurIPS 2022 paper:

["Continuous MDP Homomorphisms and Homomorphic Policy Gradient". Sahand Rezaei-Shoshtari, Rosie Zhao, Prakash Panangaden, David Meger, and Doina Precup. In Advances in Neural Information Processing Systems (NeurIPS). 2022.
](https://arxiv.org/abs/2209.07364)

DHPG simultaneously learns the MDP homomorphism map and learns the optimal policy using the 
homomorphic policy gradient theorem for continuous control problems:
<p align="center">
  <img src="figures/hpg_diagram.png" alt="HPG diagram." width="500"/>
</p>


## Setup
* Install the following libraries needed for Mujoco and DeepMind Control Suite:
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
* Install dependencies of this package:
```commandline
pip install -r requirements.txt
````

## Instructions
* This code includes our Python implementation of DHPG and all 
the baseline algorithms used in the paper: 
  * **Pixel observations:** DHPG, DBC, DeepMDP, SAC-AE, DrQ-v2.
  * **State observations:** DHPG, TD3, DDPG, SAC.
* Results were obtained on Python v3.8.10, CUDA v11.4, PyTorch v1.10.0 on 10 seeds.
* Our code will be publicly released after the review process.

### Training on Pixel Observations (Section 7.2, Appendices D.2, D.5, D.6)
* To train agents on pixel observations:
```commandline
python train.py task=pendulum_swingup agent=hpg 
```
* Available **DHPG** agents are: `hpg`, `hpg_aug`, `hpg_ind`, `hpg_ind_aug`:
  * `hpg` is the DHPG variant in which gradients of HPG and DPG are summed 
  together for a single actor update (`hpg_aug` is `hpg` with image augmentation.) 
  * `hpg_ind` is the DHPG variant in which gradients of HPG and DPG are 
   used to independently update the actor (`hpg_ind_aug` is `hpg_ind` with image augmentation.)
  * See Appendix D.5 for more information on these variants. 
* Available **baseline** agents are: `drqv2`, `dbc`, `deepmdp`, `sacae`.
  * You can run each baseline with image augmentation by simply adding `_aug` to the end
  of its name. For example, `dbc_aug` runs `dbc` with image augmentation. 
* If you do not have a CUDA device, use `device=cpu`.

### Training on State Observations (Section 7.1, Appendix D.1)
* To train agents on state observations:
```commandline
python train.py pixel_obs=false action_repeat=1 frame_stack=1 task=pendulum_swingup agent=hpg 
```
* Available **DHPG** agents are: `hpg`, `hpg_ind`.
* Available **baseline** agents are: `td3`, `sac`, `ddpg_original`, `ddpg_ours`.

### Transfer Experiments (Appendix D.3)
* To run the transfer experiments, use `python transfer.py` with the same configurations discussed above for
pixel observations, but use `cartpole_transfer`, `quadruped_transfer`, `walker_transfer`, or `hopper_transfer` 
as the `task` argument.

### Tensorboard
* To monitor results use:
```commandline
tensorboard --logdir exp
```

## Citation
If you are using our code, please cite our NeurIPS 2022 paper: 
```bib
@article{rezaei2022continuous,
  title={Continuous MDP Homomorphisms and Homomorphic Policy Gradient},
  author={Rezaei-Shoshtari, Sahand and Zhao, Rosie and Panangaden, Prakash and Meger, David and Precup, Doina},
  journal={arXiv preprint arXiv:2209.07364},
  year={2022}
}
```
