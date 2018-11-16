# QuanFaceAlignment: A bottom-up face alignment method.

By Wu Yuxiang and [Huang Bin](https://github.com/Isaver23)

## License

The code of QuanFaceAlignment is released under the MIT License.

## Recent Update

**`2018.11.13`**: We implemented [LAB](https://github.com/wywu/LAB)(A top-down method)

## Prerequisites

 - ubuntu 16.04
 - python 3.6.7
 - pytorch 0.41
 - imgaug
 - horovod
 
 ## Getting Started

---

### Installation

```shell
pip install -r requirements.txt
```

### Data preparation

 - Download [WFLW_images.tar.gz](https://drive.google.com/open?id=1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC) and [WFLW_annotations.tar.gz](https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz), extract them to ./data.
 - Run lab_crop.py to preprocess the data.
 ```shell
 python lab_crop.py --input-dir ./data/WFLW_images --output-dir ./data/WFLW_crop_256_npy --anno-path ./data/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt  --anno-save-path ./data/train.txt 
 
 python lab_crop.py --input-dir ./data/WFLW_images --output-dir ./data/WFLW_crop_256_npy --anno-path ./data/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt  --anno-save-path ./data/test.txt
 ```
 - Open ./datasets/wflw.py, change train_path and test_path to './data/train.txt' and './data/test.txt'.

### Training

 Run ./train_landmarks.py for training.
 ```shell
 python train_landmarks.py --tensorboard-path ./TB --save-params-path ./param,1 --per-batch 10 --epochs 500
 ```
