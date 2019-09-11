# SenseActExperiments
A Survey on Reproducibility by Evaluating Deep Reinforcement Learning Algorithms on Real-World Robots

## Citing

For our fork of [SenseAct](https://github.com/dti-research/SenseAct) and our replication work on the experiments please cite Lynnerup et al. (2019).

* Lynnerup, N. A., Nolling, L., Hallam, J., Hasle, R. (2019). [A Survey on Reproducibility by Evaluating Deep Reinforcement Learning Algorithms on Real-World Robots](https://arxiv.org/abs/1909.03772). In *Proceedings of the 3rd International Conference on Robot Learning - Volume 100.* (Osaka, Japan). Proceedings of Machine Learning Research (PMLR).

## TL;DR show me how to run it!

To run the experiments you need to have a working install of [Docker](https://docs.docker.com/) prior to proceeding.

1. On the host machine do:

```bash
# 1. Clone this repo (anywhere on your PC)
git clone https://github.com/dti-research/SenseActExperiments.git

# 2. Change directory into the docker folder
cd SenseActExperiments/

# 3. Start the Docker container (w/ experiment folder bind mounted into )
docker run -it --rm -v $PWD:/code -w /code dtiresearch/senseact bash
```

If you want to build our version of the [SenseAct](https://github.com/dti-research/SenseAct) framework from source, please see [INSTALL.md](INSTALL.md).

2. Inside the Docker container

```bash
# 1. Go into the folder containing the code
root@2c9828ac7a9b:/code# cd code

# 2. Run the train script given the experiment setup file
root@2c9828ac7a9b:/code/code# python3 train/train.py -f experiments/ur5/trpo_kindred_example.yaml
```

That's it folks! You should be able to obtain results very similar to ours.


## How do I navigate?

In an attempt to achieve a higher level of reproducibility and minimize bias we showcase the following project structure, which is inspired from "RE-EVALUATE: Reproducibility in Evaluating Reinforcement Learning Algorithms". The idea is to allow for "easy" addition of new experiments, algorithms, environments, and measurement metrics.

```
code
├── algorithms
├── artifacts
│   ├── logs
│   └── models
├── environments
├── experiments
├── train
└── utils
```

All code for generating reported figures and statistical inference is located in the evaluate folder, at the root of this repo.

## Setting up the offline robot simulator

1. Go to https://www.universal-robots.com/download/
1. Select robot type (CB- or e-series)
1. Choose "Software"
1. Choose "Offline Simulator"
1. Choose "Non Linux" (to get their VM)
    1. When Download completes unrar the package (Note that you with some of the packages have to do the unrar in Windows as Ubuntu does not do it correctly)
1. Load the VM with the UR simulator
1. Set the network settings to bridge
1. Start the VM
1. Inside the VM verify in the terminal that you have an IP and try to ping it from your host machine.
1. Start the offline simulator by double-clicking it on the VM desktop.
1. Simply set the IP address (in the setup.py file) to the one of your VM and you're good to go

**Warning!** You should **not** run VM on the same machine you run Docker on!
