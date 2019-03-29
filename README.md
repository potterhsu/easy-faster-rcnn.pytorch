# easy-faster-rcnn.pytorch

An easy implementation of [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) in PyTorch.


## Demo

![](images/demo-result.jpg)

1. Download checkpoint from [here](https://drive.google.com/open?id=1mQ9HV5nmGBM06mg1DjKWqBuKoipoBe5U)
1. Follow the instructions in [Setup](#setup) 2 & 3
1. Run inference script
    ```
    $ python infer.py -s=coco2017 -b=resnet101 -c=/path/to/checkpoint.pth --image_min_side=800 --image_max_side=1333 --anchor_sizes="[64, 128, 256, 512]" --rpn_post_nms_top_n=1000 /path/to/input/image.jpg /path/to/output/image.jpg
    ```

## Features

* Supports PyTorch 1.0
* Supports `PASCAL VOC 2007` and `MS COCO 2017` datasets
* Supports `ResNet-18`, `ResNet-50` and `ResNet-101` backbones (from official PyTorch model)
* Supports `ROI Pooling` and `ROI Align` pooler modes
* Supports `Multi-Batch` and `Multi-GPU` training
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
            <th>#GPUs</th>
            <th>#Batches/GPU</th>
            <th>Training Speed (FPS)</th>
            <th>Inference Speed (FPS)</th>
            <th>mAP</th>
            <th>image_min_side</th>
            <th>image_max_side</th>
            <th>anchor_ratios</th>
            <th>anchor_sizes</th>
            <th>pooler_mode</th>
            <th>rpn_pre_nms_top_n (train)</th>
            <th>rpn_post_nms_top_n (train)</th>
            <th>rpn_pre_nms_top_n (eval)</th>
            <th>rpn_post_nms_top_n (eval)</th>
            <th>anchor_smooth_l1_loss_beta</th>
            <th>proposal_smooth_l1_loss_beta</th>
            <th>batch_size</th>
            <th>learning_rate</th>
            <th>momentum</th>
            <th>weight_decay</th>
            <th>step_lr_sizes</th>
            <th>step_lr_gamma</th>
            <th>warm_up_factor</th>
            <th>warm_up_num_iters</th>
            <th>num_steps_to_finish</th>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/abs/1506.01497">
                    Original Paper
                </a>
            </td>
            <td>VGG-16</td>
            <td>Tesla K40</td>
            <td>1</td>
            <td>1</td>
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
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/ruotianluo/pytorch-faster-rcnn">
                    ruotianluo/pytorch-faster-rcnn
                </a>
            </td>
            <td>ResNet-101</td>
            <td>TITAN Xp</td>
            <td>-</td>
            <td>-</td>
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
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/jwyang/faster-rcnn.pytorch">
                    jwyang/faster-rcnn.pytorch
                </a>
            </td>
            <td>ResNet-101</td>
            <td>TITAN Xp</td>
            <td>1</td>
            <td>1</td>
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
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/open?id=1IQSqnLFKduZaYcq06lZmYd_e7qbH3cxZ">
                    Ours
                </a>
            </td>
            <td>ResNet-101</td>
            <td>GTX 1080 Ti</td>
            <td>1</td>
            <td>4</td>
            <td>7.12</td>
            <td>15.05</td>
            <td>0.7562</td>
            <td>600</td>
            <td>1000</td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td>[128, 256, 512]</td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td>300</td>
            <td>1.0</td>
            <td>1.0</td>
            <td><b>4</b></td>
            <td><b>0.004</b></td>
            <td>0.9</td>
            <td>0.0005</td>
            <td><b>[12500, 17500]</b></td>
            <td>0.1</td>
            <td>0.3333</td>
            <td>500</td>
            <td><b>22500</b></td>
        </tr>
    </table>

* MS COCO 2017

    * Train: 2017 Train drops images without any objects (117266 images)
    * Eval: 2017 Val drops images without any objects (4952 images)

    <table>
        <tr>
            <th>Implementation</th>
            <th>Backbone</th>
            <th>GPU</th>
            <th>#GPUs</th>
            <th>#Batches/GPU</th>
            <th>Training Speed (FPS)</th>
            <th>Inference Speed (FPS)</th>
            <th>AP@[.5:.95]</th>
            <th>AP@[.5]</th>
            <th>AP@[.75]</th>
            <th>AP S</th>
            <th>AP M</th>
            <th>AP L</th>
            <th>image_min_side</th>
            <th>image_max_side</th>
            <th>anchor_ratios</th>
            <th>anchor_sizes</th>
            <th>pooler_mode</th>
            <th>rpn_pre_nms_top_n (train)</th>
            <th>rpn_post_nms_top_n (train)</th>
            <th>rpn_pre_nms_top_n (eval)</th>
            <th>rpn_post_nms_top_n (eval)</th>
            <th>anchor_smooth_l1_loss_beta</th>
            <th>proposal_smooth_l1_loss_beta</th>
            <th>batch_size</th>
            <th>learning_rate</th>
            <th>momentum</th>
            <th>weight_decay</th>
            <th>step_lr_sizes</th>
            <th>step_lr_gamma</th>
            <th>warm_up_factor</th>
            <th>warm_up_num_iters</th>
            <th>num_steps_to_finish</th>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/ruotianluo/pytorch-faster-rcnn">
                    ruotianluo/pytorch-faster-rcnn
                </a>
            </td>
            <td>ResNet-101</td>
            <td>TITAN Xp</td>
            <td>-</td>
            <td>-</td>
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
                <a href="https://github.com/jwyang/faster-rcnn.pytorch">
                    jwyang/faster-rcnn.pytorch
                </a>
            </td>
            <td>ResNet-101</td>
            <td>TITAN Xp</td>
            <td>8</td>
            <td>2</td>
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
                <a href="https://drive.google.com/open?id=1d-t7sR0-9dyRQvEbisklzoOkTibGMOY8">
                    Ours
                </a>
            </td>
            <td>ResNet-101</td>
            <td>GTX 1080 Ti</td>
            <td>1</td>
            <td>2</td>
            <td>4.84</td>
            <td>8.00</td>
            <td>0.356</td>
            <td>0.562</td>
            <td>0.389</td>
            <td>0.176</td>
            <td>0.398</td>
            <td>0.511</td>
            <td><b>800</b></td>
            <td><b>1333</b></td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td><b>[64, 128, 256, 512]</b></td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td><b>1000</b></td>
            <td><b>0.1111</b></td>
            <td>1.0</td>
            <td><b>2</b></td>
            <td><b>0.0025</b></td>
            <td>0.9</td>
            <td><b>0.0001</b></td>
            <td><b>[480000, 640000]</b></td>
            <td>0.1</td>
            <td>0.3333</td>
            <td>500</td>
            <td><b>720000</b></td>
        </tr>
        <tr>
            <td>
                <a href="https://drive.google.com/open?id=1faXOZXtImWfAXzfgpE6q5-2VP2lw39q1">
                    Ours
                </a>
            </td>
            <td>ResNet-101</td>
            <td>Telsa P100</td>
            <td>4</td>
            <td>4</td>
            <td>11.64</td>
            <td>5.10</td>
            <td>0.370</td>
            <td>0.576</td>
            <td>0.403</td>
            <td>0.187</td>
            <td>0.414</td>
            <td>0.522</td>
            <td><b>800</b></td>
            <td><b>1333</b></td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td><b>[64, 128, 256, 512]</b></td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td><b>1000</b></td>
            <td><b>0.1111</b></td>
            <td>1.0</td>
            <td><b>16</b></td>
            <td><b>0.02</b></td>
            <td>0.9</td>
            <td><b>0.0001</b></td>
            <td><b>[120000, 160000]</b></td>
            <td>0.1</td>
            <td>0.3333</td>
            <td>500</td>
            <td><b>180000</b></td>
        </tr>
    </table>

* PASCAL VOC 2007 Cat Dog

    * Train: 2007 trainval drops categories other than cat and dog (750 images)
    * Eval: 2007 test drops categories other than cat and dog (728 images)

* MS COCO 2017 Person

    * Train: 2017 Train drops categories other than person (64115 images)
    * Eval: 2017 Val drops categories other than person (2693 images)

* MS COCO 2017 Car

    * Train: 2017 Train drops categories other than car (12251 images)
    * Eval: 2017 Val drops categories other than car (535 images)

* MS COCO 2017 Animal

    * Train: 2017 Train drops categories other than bird, cat, dog, horse, sheep, cow, elephant, bear, zebra and giraffe (23989 images)
    * Eval: 2017 Val drops categories other than bird, cat, dog, horse, sheep, cow, elephant, bear, zebra and giraffe (1016 images)


## Requirements

* Python 3.6
* torch 1.0
* torchvision 0.2.1
* tqdm
    ```
    $ pip install tqdm
    ```

* tensorboardX
    ```
    $ pip install tensorboardX
    ```
    
* OpenCV 3.4 (required by `infer_stream.py`)
    ```
    $ pip install opencv-python~=3.4
    ```

* websockets (required by `infer_websocket.py`)
    ```
    $ pip install websockets
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

1. Build `Non Maximum Suppression` and `ROI Align` modules (modified from [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark))

    1. Install

        ```
        $ python support/setup.py develop
        ```

    1. Uninstall

        ```
        $ python support/setup.py develop --uninstall
        ```

    1. Test

        ```
        $ python test/nms/test_nms.py
        ```

        * Result

            ![](images/test_nms.png?raw=true)

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
        $ python train.py -s=voc2007 -b=resnet101
        ```

    * To apply custom configuration (see also `train.py`)
        ```
        $ python train.py -s=voc2007 -b=resnet101 --weight_decay=0.0001
        ```

    * To apply recommended configuration (see also `scripts/`)
        ```
        $ bash ./scripts/voc2007/train-bs2.sh resnet101 /path/to/outputs/dir
        ```

1. Evaluate

    * To apply default configuration (see also `config/`)
        ```
        $ python eval.py -s=voc2007 -b=resnet101 /path/to/checkpoint.pth
        ```

    * To apply custom configuration (see also `eval.py`)
        ```
        $ python eval.py -s=voc2007 -b=resnet101 --rpn_post_nms_top_n=1000 /path/to/checkpoint.pth
        ```

    * To apply recommended configuration (see also `scripts/`)
        ```
        $ bash ./scripts/voc2007/eval.sh resnet101 /path/to/checkpoint.pth
        ```

1. Infer

    * To apply default configuration (see also `config/`)
        ```
        $ python infer.py -s=voc2007 -b=resnet101 -c=/path/to/checkpoint.pth /path/to/input/image.jpg /path/to/output/image.jpg
        ```

    * To apply custom configuration (see also `infer.py`)
        ```
        $ python infer.py -s=voc2007 -b=resnet101 -c=/path/to/checkpoint.pth -p=0.9 /path/to/input/image.jpg /path/to/output/image.jpg
        ```

    * To apply recommended configuration (see also `scripts/`)
        ```
        $ bash ./scripts/voc2007/infer.sh resnet101 /path/to/checkpoint.pth /path/to/input/image.jpg /path/to/output/image.jpg
        ```

1. Infer other sources

    * Source from stream (see also `infer_stream.py`)
        ```
        # Camera
        $ python infer_stream.py -s=voc2007 -b=resnet101 -c=/path/to/checkpoint.pth -p=0.9 0 5
        
        # Video
        $ python infer_stream.py -s=voc2007 -b=resnet101 -c=/path/to/checkpoint.pth -p=0.9 /path/to/file.mp4 5
        
        # Remote
        $ python infer_stream.py -s=voc2007 -b=resnet101 -c=/path/to/checkpoint.pth -p=0.9 rtsp://184.72.239.149/vod/mp4:BigBuckBunny_115k.mov 5
        ```
        
    * Source from websocket (see also `infer_websocket.py`)
        1. Start web server
            ```
            $ cd webapp
            $ python -m http.server 8000
            ```
            
        1. Launch service
            ```
            $ python infer_websocket.py -s=voc2007 -b=resnet101 -c=/path/to/checkpoint.pth -p=0.9
            ```
            
        1. Navigate website: `http://127.0.0.1:8000/`
        
            ![](images/web-app.jpg)
            
            > Sample video from [Pexels](https://www.pexels.com/videos/)


## Notes

* Illustration for "find labels for each `anchor_bboxes`" in `region_proposal_network.py`

    ![](images/rpn_find_labels_1.png)

    ![](images/rpn_find_labels_2.png)

* Illustration for NMS CUDA

    ![](images/nms_cuda.png)

* Plot of beta smooth L1 loss function

    ![](images/beta-smooth-l1.png)
