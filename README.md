# easy-faster-rcnn.pytorch

An easy implementation of Faster R-CNN in PyTorch.


## Demo

![](https://github.com/potterhsu/easy-faster-rcnn.pytorch/blob/master/images/inference-result.jpg?raw=true)


## Features

* Supports PyTorch 0.4.1
* Supports `PASCAL VOC 2007` and `MS COCO 2017` datasets
* Supports `VGG 16` and `ResNet 101` backbones (from official PyTorch model)
* Supports `ROI Pooling` and `ROI Align` pooling modes
* Matches the performance reported by the original paper
* It's efficient with maintainable, readable and clean code


## Benchmarking

* PASCAL VOC 2007

    * Train: 2007 trainval (5011 samples)
    * Eval: 2007 test (4952 samples)

    <table>
        <tr>
            <th>Implementation</th>
            <th>Backbone</th>
            <th>GPU</th>
            <th>Training Speed (FPS)</th>
            <th>Inference Speed (FPS)</th>
            <th>mAP</th>
        </tr>
        <tr>
            <td>Original Paper</td>
            <td>VGG-16</td>
            <td>Tesla K40</td>
            <td>N/A</td>
            <td>~ 5</td>
            <td>0.699</td>
        </tr>
        <tr>
            <td>chenyuntc/simple-faster-rcnn-pytorch</td>
            <td>VGG-16</td>
            <td>TITAN Xp</td>
            <td>~ 6.5</td>
            <td>~ 14.4</td>
            <td>0.712</td>
        </tr>
        <tr>
            <td>ruotianluo/pytorch-faster-rcnn</td>
            <td>VGG-16</td>
            <td>TITAN Xp</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>0.7122</td>
        </tr>
        <tr>
            <td>jwyang/faster-rcnn.pytorch</td>
            <td>VGG-16</td>
            <td>TITAN Xp</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>0.701</td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/drive/folders/1SKNPzSPFlLL_Y2XhRt6rA0d28Jh6Sy-4?usp=sharing">
                    Ours
                </a>
            </td>
            <td>VGG-16</td>
            <td>GTX 1080 Ti</td>
            <td>~ 6.9</td>
            <td>~ 15.5</td>
            <td>0.7013</td>
        </tr>
        <tr>
            <td>ruotianluo/pytorch-faster-rcnn</td>
            <td>ResNet-101</td>
            <td>TITAN Xp</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>0.7576</td>
        </tr>
        <tr>
            <td>jwyang/faster-rcnn.pytorch</td>
            <td>ResNet-101</td>
            <td>TITAN Xp</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>0.752</td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/drive/folders/1HCefw8-4eCC5MVz07bhROjFBvt4SKRGD?usp=sharing">
                    Ours
                </a>
            </td>
            <td>ResNet-101</td>
            <td>GTX 1080 Ti</td>
            <td>~ 5.6</td>
            <td>~ 11.7</td>
            <td>0.7538</td>
        </tr>
    </table>


* MS COCO 2017

    * Train: 2017 Train = 2015 Train + 2015 Val - 2015 Val Sample 5k (117266 samples)
    * Eval: 2017 Val = 2015 Val Sample 5k (formerly known as `minival`) (4952 samples)

    <table>
        <tr>
            <th>Implementation</th>
            <th>Backbone</th>
            <th>GPU</th>
            <th>Training Speed (FPS)</th>
            <th>Inference Speed (FPS)</th>
            <th>AP@[.5:.95]</th>
        </tr>
        <tr>
            <td>ruotianluo/pytorch-faster-rcnn</td>
            <td>VGG-16</td>
            <td>TITAN Xp</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>0.301</td>
        </tr>
        <tr>
            <td>jwyang/faster-rcnn.pytorch</td>
            <td>VGG-16</td>
            <td>TITAN Xp</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>0.292</td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/drive/folders/1_2X_Sn311f-hs9S0bVyMmeCsE9NppQHI?usp=sharing">
                    Ours
                </a>
            </td>
            <td>VGG-16</td>
            <td>GTX 1080 Ti</td>
            <td>~ 3.9</td>
            <td>~ 6.5</td>
            <td>0.287</td>
        </tr>
        <tr>
            <td>ruotianluo/pytorch-faster-rcnn</td>
            <td>ResNet-101</td>
            <td>TITAN Xp</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>0.354</td>
        </tr>
        <tr>
            <td>jwyang/faster-rcnn.pytorch</td>
            <td>ResNet-101</td>
            <td>TITAN Xp</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>0.370</td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/drive/folders/16VyI2GjLrf3uU_bhqvZpC4ZGpen0wS5d?usp=sharing">
                    Ours
                </a>
            </td>
            <td>ResNet-101</td>
            <td>GTX 1080 Ti</td>
            <td>~ 3.4</td>
            <td>~ 5.4</td>
            <td>0.352</td>
        </tr>
    </table>

## Requirements

* Python 3.6
* torch 0.4.1
* torchvision 0.2.1
* tqdm

    ```
    $ pip install tqdm
    ```


## Setup

1. Prepare data
    1. For `PASCAL VOC 2007`

        1. Download dataset

            - [Training / Validation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (5011 images)
            - [Test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) (4952 images)

        1. Extract to data folder, now your folder structure should be like:

            ```
            easy-faster-rcnn.pytorch
                - data
                    - VOCdevkit
                        - VOC2007
                            - Annotations
                                - 000001.xml
                                - 000002.xml
                                ...
                            - ImageSets
                                - Main
                                    ...
                                    test.txt
                                    ...
                                    trainval.txt
                                    ...
                            - JPEGImages
                                - 000001.jpg
                                - 000002.jpg
                                ...
                    - ...
            ```

    1. For `MS COCO 2017`

        1. Download dataset

            - [2017 Train images [18GB]](http://images.cocodataset.org/zips/train2017.zip) (118287 images)
                > COCO 2017 Train = COCO 2015 Train + COCO 2015 Val - COCO 2015 Val Sample 5k
            - [2017 Val images [1GB]](http://images.cocodataset.org/zips/val2017.zip) (5000 images)
                > COCO 2017 Val = COCO 2015 Val Sample 5k (formerly known as `minival`)
            - [2017 Train/Val annotations [241MB]](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

        1. Extract to data folder, now your folder structure should be like:

            ```
            easy-faster-rcnn.pytorch
                - data
                    - COCO
                        - annotations
                            - instances_train2017.json
                            - instances_val2017.json
                            ...
                        - train2017
                            - 000000000009.jpg
                            - 000000000025.jpg
                            ...
                        - val2017
                            - 000000000139.jpg
                            - 000000000285.jpg
                            ...
                    - ...
            ```

1. Build CUDA modules

    1. Define your CUDA architecture code

        ```
        $ export CUDA_ARCH=sm_61
        ```

        * `sm_61` is for `GTX 1080 Ti`, to see others visit [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

        * To check your GPU architecture, you might need following script to find out GPU information

            ```
            $ nvidia-smi -L
            ```

    1. Build `Non-Maximum-Suppression` module

        ```
        $ nvcc -arch=$CUDA_ARCH -c --compiler-options -fPIC -o nms/src/nms_cuda.o nms/src/nms_cuda.cu
        $ python nms/build.py
        $ python -m nms.test.test_nms
        ```

        * Result after unit testing

            ![](https://github.com/potterhsu/easy-faster-rcnn.pytorch/blob/master/images/test_nms.png?raw=true)

        * Illustration for NMS CUDA

            ![](https://github.com/potterhsu/easy-faster-rcnn.pytorch/blob/master/images/nms_cuda.png?raw=true)

    1. Build `ROI-Align` module (modified from [RoIAlign.pytorch](https://github.com/longcw/RoIAlign.pytorch))

        ```
        $ nvcc -arch=$CUDA_ARCH -c --compiler-options -fPIC -o roi/align/src/cuda/crop_and_resize_kernel.cu.o roi/align/src/cuda/crop_and_resize_kernel.cu
        $ python roi/align/build.py
        ```

1. Install `pycocotools` for `MS COCO 2017` dataset

    1. Clone and build COCO API

        ```
        $ git clone https://github.com/cocodataset/cocoapi
        $ cd cocoapi/PythonAPI
        $ make
        ```
        > It's not necessary to be under project directory

    1. Copy `pycocotools` into project

        ```
        $ cp -R pycocotools /path/to/project
        ```


## Usage

1. Train

    * To apply default configuration (see also `config/`)
        ```
        $ python train.py -s=voc2007 -b=vgg16
        ```

    * To apply recommended configuration (see also `scripts/`)
        ```
        $ bash ./scripts/voc2007/vgg16/train.sh
        ```

    * To apply custom configuration (see also `train.py`)
        ```
        $ python train.py -s=voc2007 -b=vgg16 --pooling_mode=pooling --weight_decay=0.0001
        ```

1. Evaluate

    * To apply default configuration (see also `config/`)
        ```
        $ python eval.py -s=voc2007 -b=vgg16 /path/to/checkpoint.pth
        ```

    * To apply recommended configuration (see also `scripts/`)
        ```
        $ bash ./scripts/voc2007/vgg16/eval.sh /path/to/checkpoint.pth
        ```

    * To apply custom configuration (see also `eval.py`)
        ```
        $ python eval.py -s=voc2007 -b=vgg16 --pooling_mode=pooling --rpn_post_nms_top_n=1000
        ```

1. Infer

    * To apply default configuration (see also `config/`)
        ```
        $ python infer.py -c=/path/to/checkpoint.pth -s=voc2007 -b=vgg16 /path/to/input/image.jpg /path/to/output/image.jpg
        ```

    * To apply recommended configuration (see also `scripts/`)
        ```
        $ bash ./scripts/voc2007/vgg16/infer.sh /path/to/checkpoint.pth /path/to/input/image.jpg /path/to/output/image.jpg
        ```

    * To apply custom configuration (see also `infer.py`)
        ```
        $ python infer.py -c=/path/to/checkpoint.pth -s=voc2007 -b=vgg16 -p=0.9 /path/to/input/image.jpg /path/to/output/image.jpg
        ```
