# Install

Install Docker by using their official guide: [https://docs.docker.com/install/](https://docs.docker.com/install/).

Next either:

(a) pull the build Docker image from docker hub by:

```bash
# Pull the Docker image from the Docker Hub
docker pull dtiresearch/senseact
```

(b) build the Docker image yourself, by:

```bash
# 1. Clone the repo
git clone https://github.com/dti-research/SenseAct.git

# 2. Change directory into the docker folder
cd SenseAct/docker

# 3. Build Docker image
docker build -f Dockerfile -t dtiresearch/senseact .
```

**Be Advised!** If you do not have MuJoCo installed then the instruction for installing OpenAI Baselines v.0.1.5 will give an error, but the image is still build successfully and fully functional! It is an error in the dependency packages for Baselines v.0.1.5 which should be fixed in later versions.