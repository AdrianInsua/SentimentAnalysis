# SentimentAnalysis
Spanish NLP Sentiment Analysis investigation and implementation

## Investigation
We use Tensorflow as NN backend, and keras as framework in order to get an easier implementation

### Required libraries
We have included a script to install all required libraries and jupyter-notebook (with extensions) using pip command, if your system uses pip3 edit **install.sh** and replace pip with pip3.

Execute:

`sh install.sh`

**Optional** In order to have a better jupyter experience you should enable **table of contents** extension

#### W2vec

You will need [w2vec binaries](http://crscardellino.github.io/SBWCE/) in order to get word embeddings for DL

### Tensorflow
[Tensorflow](https://www.tensorflow.org/)

To use our notebooks you can make a local installation of tensorflow or use a docker image (Follow tensorflow docs instructions).

It is not necessary to use tensorflow GPU version, but you will get better performance if your system is able to use it. see [CUDA supported devices](https://developer.nvidia.com/cuda-gpus).

#### Docker Image Recomendations

If you have problems with docker-nvidia2 installation or docker-ce installation (i.e: Ubuntu 19.04 have no stable version released of docker) we recommend to follow this [guide](https://www.pugetsystems.com/labs/hpc/How-To-Install-Docker-and-NVIDIA-Docker-on-Ubuntu-19-04-1460/)

We recommend to use the following command from project source dir:

`docker run  --runtime=nvidia -it --name tf -v $PWD:/tmp -w /tmp --rm  -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3`

This opens a bash in docker with redirection to project dir and 8888 ports open so we can use jupyter-notebook

Instal required libraries and then execute: 

`jupyter-notebook --allow-root --port=8888 --ip=0.0.0.0`
