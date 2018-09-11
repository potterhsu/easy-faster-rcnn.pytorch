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
        <th>aeroplane</th>
        <th>bicycle</th>
        <th>bird</th>
        <th>boat</th>
        <th>bottle</th>
        <th>bus</th>
        <th>car</th>
        <th>cat</th>
        <th>chair</th>
        <th>cow</th>
        <th>diningtable</th>
        <th>dog</th>
        <th>horse</th>
        <th>motorbike</th>
        <th>person</th>
        <th>pottedplant</th>
        <th>sheep</th>
        <th>sofa</th>
        <th>train</th>
        <th>tvmonitor</th>
    </tr>
    <tr>
        <td>
            <a href="https://drive.google.com/open?id=1nkKGnT8TGVPtOwICkHhSdi9gsnglHZ8N">
                VGG-16
            </a>
        </td>
        <td>~7 examples/sec</td>
        <td>~13 examples/sec</td>
        <td>0.7029</td>
        <td>0.7133</td>
        <td>0.7814</td>
        <td>0.6815</td>
        <td>0.5606</td>
        <td>0.5272</td>
        <td>0.8188</td>
        <td>0.7919</td>
        <td>0.8296</td>
        <td>0.4973</td>
        <td>0.7729</td>
        <td>0.6799</td>
        <td>0.7945</td>
        <td>0.8018</td>
        <td>0.7647</td>
        <td>0.7655</td>
        <td>0.4244</td>
        <td>0.7152</td>
        <td>0.6748</td>
        <td>0.7480</td>
        <td>0.7157</td>
    </tr>
    <tr style="color: gray;">
        <td>Resnet-101 (freeze 0~4)</td>
        <td>~5.4 examples/sec</td>
        <td>~11 examples/sec</td>
        <td>0.7496</td>
        <td>0.7629</td>
        <td>0.8362</td>
        <td>0.7708</td>
        <td>0.6364</td>
        <td>0.5557</td>
        <td>0.8110</td>
        <td>0.8305</td>
        <td>0.8732</td>
        <td>0.5944</td>
        <td>0.8081</td>
        <td>0.7139</td>
        <td>0.8726</td>
        <td>0.8588</td>
        <td>0.7765</td>
        <td>0.7898</td>
        <td>0.4726</td>
        <td>0.7518</td>
        <td>0.7405</td>
        <td>0.7925</td>
        <td>0.7444</td>
    </tr>
    <tr style="color: gray;">
        <td>Resnet-101 (freeze 0~5)</td>
        <td>~5.7 examples/sec</td>
        <td>~11 examples/sec</td>
        <td>0.7466</td>
        <td>0.7734</td>
        <td>0.8276</td>
        <td>0.7855</td>
        <td>0.5990</td>
        <td>0.5661</td>
        <td>0.8177</td>
        <td>0.8161</td>
        <td>0.8754</td>
        <td>0.5703</td>
        <td>0.7919</td>
        <td>0.6990</td>
        <td>0.8717</td>
        <td>0.8510</td>
        <td>0.7936</td>
        <td>0.7875</td>
        <td>0.4793</td>
        <td>0.7537</td>
        <td>0.7225</td>
        <td>0.7963</td>
        <td>0.7536</td>
    </tr>
    <tr>
        <td>
            <a href="https://drive.google.com/open?id=1t-lv8bNcPnoeVru3PD3BLLMmFHsUY8u9">
                Resnet-101 (freeze 0~6)
            </a>
        </td>
        <td>~7.5 examples/sec</td>
        <td>~11 examples/sec</td>
        <td>0.7523</td>
        <td>0.7555</td>
        <td>0.8390</td>
        <td>0.7861</td>
        <td>0.6063</td>
        <td>0.5396</td>
        <td>0.8209</td>
        <td>0.8000</td>
        <td>0.8970</td>
        <td>0.5792</td>
        <td>0.8386</td>
        <td>0.6823</td>
        <td>0.8907</td>
        <td>0.8757</td>
        <td>0.8222</td>
        <td>0.7750</td>
        <td>0.4837</td>
        <td>0.7524</td>
        <td>0.7513</td>
        <td>0.7864</td>
        <td>0.7635</td>
    </tr>
    <tr style="color: gray;">
        <td>Resnet-101 (freeze 0~7)</td>
        <td>~8.4 examples/sec</td>
        <td>~11 examples/sec</td>
        <td>0.6983</td>
        <td>0.7227</td>
        <td>0.7669</td>
        <td>0.7342</td>
        <td>0.5541</td>
        <td>0.4548</td>
        <td>0.7556</td>
        <td>0.7581</td>
        <td>0.8709</td>
        <td>0.5033</td>
        <td>0.7648</td>
        <td>0.6589</td>
        <td>0.8536</td>
        <td>0.8374</td>
        <td>0.7584</td>
        <td>0.7104</td>
        <td>0.4013</td>
        <td>0.6728</td>
        <td>0.6722</td>
        <td>0.7803</td>
        <td>0.7359</td>
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
    > sm_61 is for GTX-1080-Ti, to see others, visit [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

    > Try to rebuild module if unit test fails

    * result after unit testing
    
        ![](https://github.com/potterhsu/easy-faster-rcnn.pytorch/blob/master/images/test_nms.png?raw=true)
    
    * illustration for NMS CUDA
    
        ![](https://github.com/potterhsu/easy-faster-rcnn.pytorch/blob/master/images/nms_cuda.png?raw=true)    


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
