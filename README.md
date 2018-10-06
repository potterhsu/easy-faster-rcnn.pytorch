# easy-faster-rcnn.pytorch

An easy implementation of Faster R-CNN in PyTorch.


## Demo

![](https://github.com/potterhsu/easy-faster-rcnn.pytorch/blob/master/images/inference-result.jpg?raw=true)


## Results

#### VOC 2007

<table>
    <tr>
        <th>Model</th>
        <th>Training Speed</th>
        <th>Inference Speed</th>
        <th>mAP</th>
    </tr>
    <tr>
        <td>
            <a href="https://drive.google.com/open?id=1nkKGnT8TGVPtOwICkHhSdi9gsnglHZ8N">
                VGG-16
            </a>
        </td>
        <td>~7 examples/sec</td>
        <td>~13 examples/sec</td>
        <td>0.7015</td>
    </tr>
    <tr style="color: gray;">
        <td>ResNet-101 (freeze 0~4)</td>
        <td>~5.4 examples/sec</td>
        <td>~11 examples/sec</td>
        <td>0.7496</td>
    </tr>
    <tr style="color: gray;">
        <td>ResNet-101 (freeze 0~5)</td>
        <td>~5.7 examples/sec</td>
        <td>~11 examples/sec</td>
        <td>0.7466</td>
    </tr>
    <tr>
        <td>
            <a href="https://drive.google.com/open?id=1t-lv8bNcPnoeVru3PD3BLLMmFHsUY8u9">
                ResNet-101 (freeze 0~6)
            </a>
        </td>
        <td>~7.5 examples/sec</td>
        <td>~11 examples/sec</td>
        <td>0.7523</td>
    </tr>
    <tr style="color: gray;">
        <td>ResNet-101 (freeze 0~7)</td>
        <td>~8.4 examples/sec</td>
        <td>~11 examples/sec</td>
        <td>0.6983</td>
    </tr>
</table>

> All the experiments above are running on GTX-1080-Ti


## Requirements

* Python 3.6
* torch 0.4.1
* torchvision 0.2.1
* tqdm

    ```
    $ pip install tqdm
    ```

## Setup

1. Download VOC 2007 Dataset

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
    ```

1. Build non-maximum-suppression module

    ```
    $ nvcc -arch=sm_61 -c --compiler-options -fPIC -o nms/src/nms_cuda.o nms/src/nms_cuda.cu
    $ python nms/build.py
    $ python -m nms.test.test_nms
    ```
    > sm_61 is for GTX-1080-Ti, to see others visit [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

    * Result after unit testing
    
        ![](https://github.com/potterhsu/easy-faster-rcnn.pytorch/blob/master/images/test_nms.png?raw=true)
    
    * Illustration for NMS CUDA
    
        ![](https://github.com/potterhsu/easy-faster-rcnn.pytorch/blob/master/images/nms_cuda.png?raw=true)    

    * Unit test failed? Try to:

        * rebuild module

        * check your GPU architecture, you might need following script to find out GPU information

            ```
            $ nvidia-smi -L
            ```


## Usage

1. Train

    ```
    $ python train.py -b=vgg16 -d=./data -c=./checkpoints
    ```

1. Evaluate

    ```
    $ python eval.py ./checkpoints/model-100.pth -b=vgg16 -d=./data -r=./results
    ```

1. Clean

    ```
    $ rm -rf ./checkpoints
    $ rm -rf ./results
    ```

1. Infer

    ```
    $ python infer.py input-image.jpg output-image.jpg -c=./checkpoints/model-100.pth -b=vgg16
    ```
