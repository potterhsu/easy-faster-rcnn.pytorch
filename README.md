# easy-faster-rcnn.pytorch

An easy implementation of [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) in PyTorch.


## Demo

![](images/inference-result.jpg?raw=true)


## Features

* Supports PyTorch 0.4.1
* Supports `PASCAL VOC 2007` and `MS COCO 2017` datasets
* Supports `VGG-16`, `ResNet-18`, `ResNet-50` and `ResNet-101` backbones (from official PyTorch model)
* Supports `ROI Pooling` and `ROI Align` pooling modes
* Matches the performance reported by the original paper
* It's efficient with maintainable, readable and clean code


## Benchmarking

* PASCAL VOC 2007

    * Train: 2007 trainval (5011 images)
    * Eval: 2007 test (4952 images)

    <table>
        <tr>
            <th>Implementation</th>
            <th>Backbone</th>
            <th>GPU</th>
            <th>Training Speed (FPS)</th>
            <th>Inference Speed (FPS)</th>
            <th>mAP</th>
            <th>image_min_side</th>
            <th>image_max_side</th>
            <th>anchor_ratios</th>
            <th>anchor_sizes</th>
            <th>pooling_mode</th>
            <th>rpn_pre_nms_top_n (train)</th>
            <th>rpn_post_nms_top_n (train)</th>
            <th>rpn_pre_nms_top_n (eval)</th>
            <th>rpn_post_nms_top_n (eval)</th>
            <th>learning_rate</th>
            <th>momentum</th>
            <th>weight_decay</th>
            <th>step_lr_size</th>
            <th>step_lr_gamma</th>
            <th>num_steps_to_finish</th>
        </tr>
        <tr>
            <td>Original Paper</td>
            <td>VGG-16</td>
            <td>Tesla K40</td>
            <td>-</td>
            <td>~ 5</td>
            <td>0.699</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>chenyuntc/simple-faster-rcnn-pytorch</td>
            <td>VGG-16</td>
            <td>TITAN Xp</td>
            <td>~ 6.5</td>
            <td>~ 14.4</td>
            <td>0.712</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>ruotianluo/pytorch-faster-rcnn</td>
            <td>VGG-16</td>
            <td>TITAN Xp</td>
            <td>-</td>
            <td>-</td>
            <td>0.7122</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>jwyang/faster-rcnn.pytorch</td>
            <td>VGG-16</td>
            <td>TITAN Xp</td>
            <td>-</td>
            <td>-</td>
            <td>0.701</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/open?id=1SKNPzSPFlLL_Y2XhRt6rA0d28Jh6Sy-4">
                    Ours
                </a>
            </td>
            <td>VGG-16</td>
            <td>GTX 1080 Ti</td>
            <td>~ 7.6</td>
            <td>~ 17.6</td>
            <td>0.7013</td>
            <td>600</td>
            <td>1000</td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td>[128, 256, 512]</td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td>300</td>
            <td>0.001</td>
            <td>0.9</td>
            <td>0.0005</td>
            <td>50000</td>
            <td>0.1</td>
            <td>70000</td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/open?id=1Yv2C3htLpG-KkICyQ0SbeyIOyYJsXMHq">
                    Ours
                </a>
            </td>
            <td>ResNet-18</td>
            <td>GTX 1080 Ti</td>
            <td>~ 19.4</td>
            <td>~ 38.7</td>
            <td>0.6769</td>
            <td>600</td>
            <td>1000</td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td>[128, 256, 512]</td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td>300</td>
            <td>0.001</td>
            <td>0.9</td>
            <td>0.0005</td>
            <td>50000</td>
            <td>0.1</td>
            <td>70000</td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/open?id=1qUhPlIam2pl9oWwQWDa9V_YEOr5sOUYb">
                    Ours
                </a>
            </td>
            <td>ResNet-50</td>
            <td>GTX 1080 Ti</td>
            <td>~ 8.7</td>
            <td>~ 22.4</td>
            <td>0.7423</td>
            <td>600</td>
            <td>1000</td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td>[128, 256, 512]</td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td>300</td>
            <td>0.001</td>
            <td>0.9</td>
            <td>0.0005</td>
            <td>50000</td>
            <td>0.1</td>
            <td>70000</td>
        </tr>
        <tr>
            <td>ruotianluo/pytorch-faster-rcnn</td>
            <td>ResNet-101</td>
            <td>TITAN Xp</td>
            <td>-</td>
            <td>-</td>
            <td>0.7576</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>jwyang/faster-rcnn.pytorch</td>
            <td>ResNet-101</td>
            <td>TITAN Xp</td>
            <td>-</td>
            <td>-</td>
            <td>0.752</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/open?id=1HCefw8-4eCC5MVz07bhROjFBvt4SKRGD">
                    Ours
                </a>
            </td>
            <td>ResNet-101</td>
            <td>GTX 1080 Ti</td>
            <td>~ 5.6</td>
            <td>~ 11.8</td>
            <td>0.7538</td>
            <td>600</td>
            <td>1000</td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td>[128, 256, 512]</td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td>300</td>
            <td>0.001</td>
            <td>0.9</td>
            <td>0.0005</td>
            <td>50000</td>
            <td>0.1</td>
            <td>70000</td>
        </tr>
    </table>

    > Scroll to right for more configurations

* MS COCO 2017

    * Train: 2017 Train drops images without any objects (117266 images)
    * Eval: 2017 Val drops images without any objects (4952 images)

    <table>
        <tr>
            <th>Implementation</th>
            <th>Backbone</th>
            <th>GPU</th>
            <th>Training Speed (FPS)</th>
            <th>Inference Speed (FPS)</th>
            <th>AP@[.5:.95]</th>
            <th>image_min_side</th>
            <th>image_max_side</th>
            <th>anchor_ratios</th>
            <th>anchor_sizes</th>
            <th>pooling_mode</th>
            <th>rpn_pre_nms_top_n (train)</th>
            <th>rpn_post_nms_top_n (train)</th>
            <th>rpn_pre_nms_top_n (eval)</th>
            <th>rpn_post_nms_top_n (eval)</th>
            <th>learning_rate</th>
            <th>momentum</th>
            <th>weight_decay</th>
            <th>step_lr_size</th>
            <th>step_lr_gamma</th>
            <th>num_steps_to_finish</th>
        </tr>
        <tr>
            <td>ruotianluo/pytorch-faster-rcnn</td>
            <td>VGG-16</td>
            <td>TITAN Xp</td>
            <td>-</td>
            <td>-</td>
            <td>0.301</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>jwyang/faster-rcnn.pytorch</td>
            <td>VGG-16</td>
            <td>TITAN Xp</td>
            <td>-</td>
            <td>-</td>
            <td>0.292</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/open?id=1_2X_Sn311f-hs9S0bVyMmeCsE9NppQHI">
                    Ours
                </a>
            </td>
            <td>VGG-16</td>
            <td>GTX 1080 Ti</td>
            <td>~ 5.1</td>
            <td>~ 8.9</td>
            <td>0.287</td>
            <td><b>800</b></td>
            <td><b>1333</b></td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td><b>[64, 128, 256, 512]</b></td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td><b>1000</b></td>
            <td>0.001</td>
            <td>0.9</td>
            <td><b>0.0001</b></td>
            <td><b>900000</b></td>
            <td>0.1</td>
            <td><b>1200000</b></td>
        </tr>
        <tr>
            <td>ruotianluo/pytorch-faster-rcnn</td>
            <td>ResNet-101</td>
            <td>TITAN Xp</td>
            <td>-</td>
            <td>-</td>
            <td>0.354</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>jwyang/faster-rcnn.pytorch</td>
            <td>ResNet-101</td>
            <td>TITAN Xp</td>
            <td>-</td>
            <td>-</td>
            <td>0.370</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/open?id=16VyI2GjLrf3uU_bhqvZpC4ZGpen0wS5d">
                    Ours
                </a>
            </td>
            <td>ResNet-101</td>
            <td>GTX 1080 Ti</td>
            <td>~ 4.7</td>
            <td>~ 7.8</td>
            <td>0.352</td>
            <td><b>800</b></td>
            <td><b>1333</b></td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td><b>[64, 128, 256, 512]</b></td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td><b>1000</b></td>
            <td>0.001</td>
            <td>0.9</td>
            <td><b>0.0001</b></td>
            <td><b>900000</b></td>
            <td>0.1</td>
            <td><b>1200000</b></td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/open?id=15GsaYxwpo-4mWKxPIkW_C-GXeeKmNNoz">
                    Ours
                </a>
            </td>
            <td>ResNet-101</td>
            <td>GTX 1080 Ti</td>
            <td>~ 4.5</td>
            <td>~ 7.5</td>
            <td>0.358</td>
            <td><b>800</b></td>
            <td><b>1333</b></td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td><b>[32, 64, 128, 256, 512]</b></td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td><b>1000</b></td>
            <td>0.001</td>
            <td>0.9</td>
            <td><b>0.0001</b></td>
            <td><b>900000</b></td>
            <td>0.1</td>
            <td><b>1200000</b></td>
        </tr>
    </table>
    
    > Scroll to right for more configurations

* PASCAL VOC 2007 Cat Dog

    * Train: 2007 trainval drops categories other than cat and dog (750 images)
    * Eval: 2007 test drops categories other than cat and dog (728 images)

    <table>
        <tr>
            <th>Implementation</th>
            <th>Backbone</th>
            <th>GPU</th>
            <th>Training Speed (FPS)</th>
            <th>Inference Speed (FPS)</th>
            <th>mAP</th>
            <th>image_min_side</th>
            <th>image_max_side</th>
            <th>anchor_ratios</th>
            <th>anchor_sizes</th>
            <th>pooling_mode</th>
            <th>rpn_pre_nms_top_n (train)</th>
            <th>rpn_post_nms_top_n (train)</th>
            <th>rpn_pre_nms_top_n (eval)</th>
            <th>rpn_post_nms_top_n (eval)</th>
            <th>learning_rate</th>
            <th>momentum</th>
            <th>weight_decay</th>
            <th>step_lr_size</th>
            <th>step_lr_gamma</th>
            <th>num_steps_to_finish</th>
        </tr>
        <tr>
            <td>Ours</td>
            <td>ResNet-18</td>
            <td>GTX 1080 Ti</td>
            <td>~ 19.4</td>
            <td>~ 56.2</td>
            <td>0.3776</td>
            <td>600</td>
            <td>1000</td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td>[128, 256, 512]</td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td>300</td>
            <td>0.001</td>
            <td>0.9</td>
            <td>0.0005</td>
            <td><b>700</b></td>
            <td>0.1</td>
            <td><b>1000</b></td>
        </tr>
        <tr>
            <td>Ours</td>
            <td>ResNet-18</td>
            <td>GTX 1080 Ti</td>
            <td>~ 19.4</td>
            <td>~ 56.2</td>
            <td>0.6175</td>
            <td>600</td>
            <td>1000</td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td>[128, 256, 512]</td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td>300</td>
            <td>0.001</td>
            <td>0.9</td>
            <td>0.0005</td>
            <td><b>2000</b></td>
            <td>0.1</td>
            <td><b>3000</b></td>
        </tr>
        <tr>
            <td>Ours</td>
            <td>ResNet-18</td>
            <td>GTX 1080 Ti</td>
            <td>~ 19.4</td>
            <td>~ 56.2</td>
            <td>0.7639</td>
            <td>600</td>
            <td>1000</td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td>[128, 256, 512]</td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td>300</td>
            <td>0.001</td>
            <td>0.9</td>
            <td>0.0005</td>
            <td><b>7000</b></td>
            <td>0.1</td>
            <td><b>10000</b></td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/open?id=15PsO7mNtBcGeisymV1651W3zlpiH20wg">
                    Ours
                </a>
            </td>
            <td>ResNet-18</td>
            <td>GTX 1080 Ti</td>
            <td>~ 19.4</td>
            <td>~ 56.2</td>
            <td>0.8476|0.7850 (60k|70k)</td>
            <td>600</td>
            <td>1000</td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td>[128, 256, 512]</td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td>300</td>
            <td>0.001</td>
            <td>0.9</td>
            <td>0.0005</td>
            <td>50000</td>
            <td>0.1</td>
            <td>70000</td>
        </tr>
    </table>

    > Scroll to right for more configurations

* MS COCO 2017 Person

    * Train: 2017 Train drops categories other than person (64115 images)
    * Eval: 2017 Val drops categories other than person (2693 images)

* MS COCO 2017 Car

    * Train: 2017 Train drops categories other than car (12251 images)
    * Eval: 2017 Val drops categories other than car (535 images)


## Requirements

* Python 3.6
* torch 0.4.1
* torchvision 0.2.1
* tqdm

    ```
    $ pip install tqdm
    ```

* tensorboardX

    ```
    $ pip install tensorboardX
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

            ![](images/test_nms.png?raw=true)

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

    1. If an error with message `pycocotools/_mask.c: No such file or directory` has occurred, please install `cython` and try again

        ```
        $ pip install cython
        ```

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
        $ python eval.py -s=voc2007 -b=vgg16 --pooling_mode=pooling --rpn_post_nms_top_n=1000 /path/to/checkpoint.pth
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


## Notes

* Illustration for "find labels for each `anchor_bboxes`" in `region_proposal_network.py`

    ![](images/rpn_find_labels_1.png)

    ![](images/rpn_find_labels_2.png)

* Illustration for NMS CUDA

    ![](images/nms_cuda.png)
