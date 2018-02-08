# Introduce

Here are some python files based on Tensorflow platform. I have tested on v1.4.1 and it works well. GPU is preferred. 

## net.py
This file build a network merely using face images. The structure is same to that in the paper. 

```
conv1 kernel size 15*15 filter numebr 48 stride 1
pool1 average pooling 2*2 stride 2
conv2 kernel size 5*4 filter number 256 stride 1
pool2 average pooling 2*2 stride 2
conv3 kernel size 7*5 filter number 1024 stride 1
fc1 1024 * 1024
fc2 1024 * 2048
fc3 2048 * 6
```

After training, save the model in the directory.

## net-audio.py

The file combine facial images and audio feature, fine-tune the pretrained model.


## data.py

load images and preprocess and genreate next batch.

## data_face_audio.py

load images and audios, concatate images and five audio features(same person) and generate next batch.


# Reference
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
