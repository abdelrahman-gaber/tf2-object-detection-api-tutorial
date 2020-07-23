# Tensorflow 2 Object Detection API Tutorial 

[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)

### Introduction


With the [announcement](https://blog.tensorflow.org/2020/07/tensorflow-2-meets-object-detection-api.html) that Object Detection API is now compatible with Tensorflow 2, I tried to test the new models published in the [TF2 model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), and train them with my custom data. However, I have faced some problems as the scripts I have for Tensorflow 1 is not working with Tensorflow 2 (which is not surprising!), in addition to having very poor documentation and tutorials from tensorflow models repo. Therefore, in this repo I am sharing my experience, in addition to providing clean codes to run the inference and training object detection models using Tensorflow 2.
 
This tutorial should be useful for those who has experience with the API but cannot find clear documentation or examples for the new changes to use it with Tensorflow 2. However, I will add all the details and working examples for the new comers who are trying to use the object detection api for the first time, so hopefully this tutorial will make it easy for beginners to get started and run their object detection models easily.


### Roadmap

This tutorial should take you from installation, to running pre-trained detection model, and training/evaluation your models with a custom dataset.

1. [Installation](#installation)
2. Inference with pre-trained models
3. Preparing your custom dataset for training 
4. Training with your custom data, and exporting trained models for inference

### Installation

The examples in this repo is tested with python 3.6 and Tensorflow 2.2.0, but it is expected to work with other Tensorflow 2.x versions with python version 3.5 or higher.

It is recommended to install [anaconda](https://www.anaconda.com/products/individual) and create new environment for your project:

```
# create new environment
conda create --name py36-tf2 python=3.6

# activate your environment before installation or running your scripts 
conda activate py36-tf2
``` 

You need first to install tensorflow 2, either with GPU or CPU only support (slow). For Installation with GPU support, you need to have CUDA 10.1 with CUDNN 7.6 to use Tensorflow 2.2.0. You can check the compatible versions of any tensorflow version with cuda and cudnn versions from [here](https://www.tensorflow.org/install/source#tested_build_configurations).

```
# if you have NVIDIA GPU with cuda 10.1 and cudnn 7.6
pip install tensorflow-gpu==2.2.0
```

A great feature of Anaconda is that it can automatically install a local version of cudatoolkit that is compatible with your tensorflow version (But you should have the proper nvidia gpu drivers installed).

```
# installation from anaconda along with cudatoolkit
conda install -c anaconda tensorflow-gpu==2.2.0

# or to install latest version of tensorflow, just type
conda install -c anaconda tensorflow-gpu
```

for CPU only support:

```
# CPU only support (slow)
pip install tensorflow==2.2.0
```

After that, you should install the object detection api itself, which became much easier now after the latest update. The official installation instructions can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md), but I will add the instruction to install it as a python package.


Clone the TensorFlow Models repository:

```
git clone https://github.com/tensorflow/models.git
```

Make sure you have [protobuf compiler](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager) version >= 3.0, by typing `protoc --version`, or install it on Ubuntu by typing `apt install protobuf-compiler`  


Then proceed to the python package installation as follows:

```
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

The previous commands installs the object detection api as a python package that will be available in your virtual environmnet (if you created one), and will automatically install all required dependencies if not found.

Finally, to test that your installation is correct, type the following command: 

```
# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```


### Inference with pre-trained models

TODO


### Preparing your custom dataset for training

TODO

### Training with your custom data

TODO




