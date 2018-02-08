# Introduce

The trainset directory:

```
Friends.S01E03/
	-face-images/
		- chandler/
		- joey/
		- monica/
		- phoebe/
		- rachel/
		- ross/

	-speaking-audio/
		- chandler.wav
		- joey.wav
		- monica.wav
		- phoebe.wav
		- rachel.wav
		- ross.wav
Friends.S04E04/
...
Friends.S05E05/
Friends.S07E07/
Friends.S10E15/
```


## merge_audio_file_friends:
This is a function that merge different audio files(same person) into one file, specificcally this file is designed for Friends Series. You can modify it as you want.

## gen_audio_data_friends:
This is a function that extract MFCC features. The sliding window size is 20ms and a frame shift of 10ms are used. We then select mean and standard deviation of 25D MFCCs, and standard deviation of 2- MFCCs. The hyper parameters are the same as those mentioned in the original paper.

# Reference
https://bitbucket.org/herohuyongtao/mm15-speaker-naming

If you find this repository is helpful, please cite this paper:

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
