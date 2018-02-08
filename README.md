# Introduce 

This repository reproduce the paper "deep multimodal speaker naming", published in ACM MM 15. Thanks for the author shared database with me. The author implemented it in Matlab(really nice work). Considering most of deep learning platform are avaiable and convenient to make it. I rewrote it in Python on Tensorflow Platform.

## Requirement

- Tensorflow 1.4.1
- Python 2.7
- CUDA(optional)
- Matlab
- OpenCV

## Usage

### step 1
unzip matlab toolbox and addpath in matlab
```
unzip mirtoolbox17zip.zip -d $MATLAB/toolbox 
```

### step 2

run code in ./matlab and extract audio feature. Before you run them, you should modify the file location in those codes.

```
merge_audio_file_friends
gen_audio_data_friends
```

### step 3

run net.py to train faca network. After training, run net-audio.py to fine-tune the pretrain model.
```
python net.py
python net-audio.py
```
Take notices of the friend-name.txt. This file save the order of face's name as same as matlab list. Make sure the audio feature and image attribute to the same person. 

### Results

The accuracy of Friends Series S05E05.

|                  | Accuracy in Paper | Accuracy in reproduction |
| :--------------: | :---------------: | :----------------------: |
|    face model    |       86.7%       |          87.41%          |
| face-audio model |       88.5%       |         88.335%          |

With additional audio information, the accuracy improve by 1% in my reproduction code. 

# Reference

If you find this code is helpful, please cite this paper.

```
@inproceedings{hu2015deep,
title={{Deep Multimodal Speaker Naming}},
author={Hu, Yongtao and Ren, Jimmy SJ. and Dai, Jingwen and Yuan, Chang and Xu, Li and Wang, Wenping},
booktitle={Proceedings of the 23rd Annual ACM International Conference on Multimedia},
pages={1107--1110},
year={2015},
organization={ACM}
}
```
