# Tensorflow 2 Object Detection API Tutorial 

[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)

### Introduction


With the [announcement](https://blog.tensorflow.org/2020/07/tensorflow-2-meets-object-detection-api.html) that [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) is now compatible with Tensorflow 2, I tried to test the new models published in the [TF2 model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), and train them with my custom data.
However, I have faced some problems as the scripts I have for Tensorflow 1 is not working with Tensorflow 2 (which is not surprising!), in addition to having very poor documentation and tutorials from tensorflow models repo.
In this repo I am sharing my experience, in addition to providing clean codes to run the inference and training object detection models using Tensorflow 2.
 
This tutorial should be useful for those who have experience with the API but cannot find clear examples for the new changes to use it with Tensorflow 2.
However, I will add all the details and working examples for the new comers who are trying to use the object detection api for the first time, so hopefully this tutorial will make it easy for beginners to get started and run their object detection models easily.


### Roadmap

This tutorial should take you from installation, to running pre-trained detection model, and training/evaluation your models with a custom dataset.

1. [Installation](#installation)
2. [Inference with pre-trained models](#inference-with-pre-trained-models)
3. Preparing your custom dataset for training 
4. Training with your custom data, and exporting trained models for inference

### Installation

The examples in this repo is tested with python 3.6 and Tensorflow 2.2.0, but it is expected to work with other Tensorflow 2.x versions with python version 3.5 or higher.

It is recommended to install [anaconda](https://www.anaconda.com/products/individual) and create new [environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for your projects that use the installed packages:

```bash
# create new environment
conda create --name py36-tf2 python=3.6

# activate your environment before installation or running your scripts 
conda activate py36-tf2
``` 

You need first to install tensorflow 2, either with GPU or CPU only support (slow). For Installation with GPU support, you need to have CUDA 10.1 with CUDNN 7.6 to use Tensorflow 2.2.0. You can check the compatible versions of any tensorflow version with cuda and cudnn versions from [here](https://www.tensorflow.org/install/source#tested_build_configurations).

```bash
# if you have NVIDIA GPU with cuda 10.1 and cudnn 7.6
pip install tensorflow-gpu==2.2.0
```

A great feature of Anaconda is that it can automatically install a local version of cudatoolkit that is compatible with your tensorflow version (But you should have the proper nvidia gpu drivers installed).

```bash
# installation from anaconda along with cudatoolkit (tf2 version 2.2.0)
conda install -c anaconda tensorflow-gpu==2.2.0

# or to install latest version of tensorflow, just type
conda install -c anaconda tensorflow-gpu
```

for CPU only support:

```bash
# CPU only support (slow)
pip install tensorflow==2.2.0
```

After that, you should install the object detection api itself, which became much easier now after the latest update.
The official installation instructions can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md), but I will add here the instruction to install it as a python package.


Clone the TensorFlow Models repository:

```bash
git clone https://github.com/tensorflow/models.git
```

Make sure you have [protobuf compiler](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager) version >= 3.0, by typing `protoc --version`, or install it on Ubuntu by typing `apt install protobuf-compiler`  


Then proceed to the python package installation as follows:

```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

The previous commands installs the object detection api as a python package that will be available in your virtual environmnet (if you created one), and will automatically install all required dependencies if not already installed.

Finally, to test that your installation is correct, type the following command: 

```bash
# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```

For more installation options, please refer to the original [installation guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md).


### Inference with pre-trained models

To go through the tutorial, clone this repo and follow the instructions step by step. 

```bash
git clone https://github.com/abdelrahman-gaber/tf2-object-detection-api-tutorial.git
```


To get started with the Object Detection API with TF2, let's download one of the models pre-trained with coco dataset from the [tf2 detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), and use it for inference.

You can download any of the models from the table in the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), and place it in the [models/](models) directory. For example, let's download the EfficientDet D0 model.
 
```bash
cd models/
# download the model
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
# extract the downloaded file
tar -xzvf efficientdet_d0_coco17_tpu-32.tar.gz
```

In the tensorflow object detection repo, they provide a tutorial for inference in this [notebook](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_from_saved_model_tf2_colab.ipynb), but it is not so clean and needs many improvements.
Therefore, I have created a [class for object detection inference](detector.py), along with an [example script](detect_objects.py) to use this class to run the inference with input images, or from a video.   

I encourage you to have a look at the [class file](detector.py) and the [example script](detect_objects.py) and adapt it to your application. But let's first see how to use it to get the inference running with the EfficientDet D0 model we just downloaded.  

We can provide some argument when running the scripts, first the `--model_path` argument for the path of the trained model, and the `--path_to_labelmap` for the labelmap file of your dataset (here we use the one for coco dataset).
To run the detection with set of images, provide a path to the folder containing the images in the argument `--images_dir`.

```bash
python detect_objects.py --model_path models/efficientdet_d0_coco17_tpu-32/saved_model --path_to_labelmap models/mscoco_label_map.pbtxt --images_dir data/samples/images/
```

Sample output from the detection with EfficientDet D0 model:

![alt-txt-1](data/output/1.jpg)  ![alt-txt-2](data/output/4.jpg)


You can also select set of classes to be detected by passing their labels to the argument `--class_ids` as a string with the "," delimiter. For example, using `--class_ids "1,3" ` will do detection for the classes "person" and "car" only as they have id 1 and 3 respectively (you can check the id and labels from the [coco labelmap](models/mscoco_label_map.pbtxt)). Not using this argument will lead to detecting all objects in the provided labelmap.

Let's use video input by enabling the flag `--video_input`, in addition to detecting only people by passing id 1 to the `--class_ids` argument. The video used for testing is downloaded from [here](https://www.youtube.com/watch?v=pk96gqasGBQ).

```bash
python detect_objects.py --video_input --class_ids "1" --threshold 0.3  --video_path data/samples/pedestrian_test.mp4 --model_path models/efficientdet_d0_coco17_tpu-32/saved_model --path_to_labelmap models/mscoco_label_map.pbtxt
```



### Preparing your custom dataset for training

TODO

### Training with your custom data

TODO




