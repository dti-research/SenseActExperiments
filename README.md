# SenseActExperiments
A Survey on Reproducibility and Measurement Metrics by Evaluating Deep Reinforcement Learning Baseline Algorithms on Real-World Robots

## TL;DR, Show me how to run it!

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

In an attempt to achieve a higher level of reproducibility and minimize bias we propose the following project structure, which is inspired from "RE-EVALUATE: Reproducibility in Evaluating Reinforcement Learning Algorithms". The idea is to allow for "easy" addition of new experiments, algorithms, environments, and measurement metrics.

```
.
├── algorithms
├── artifacts
│   ├── logs
│   └── models
├── environments
├── experiments
├── train
└── utils
```

## Adding new experiments, algorithms, environments or measurement metrics?

1. Experiments

To add a new experiment, simply copy one of the existing YAML experiment configuration files in `code/experiments` and alter it to your needs. Change the algorithm, environment or tune the hyperparameters from within this file.

2. Algorithms
3. Environments
4. Measurement Metrics