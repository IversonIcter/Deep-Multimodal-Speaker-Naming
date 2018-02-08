# 2018.01.26 17:02:01 +08
# Author: Wang Yongjie
# Email: wangyongjie@ict.ac.cn
import cv2
import sys
import os
import random
import h5py
import pandas as pd
import numpy as np

class preprocess(object):
    """
    generate image batches
    """

    def __init__(self, path, batch_size, class_num, filename, audio_file):
        """
        init class parameter
        path:               image directory
        batch_size:         batch_size
        """
        self.path = path
        self.batch_size = batch_size
        self.class_num = class_num
        self.filename = filename
        self.audio_file = audio_file
        self.start = 0
        self.end = batch_size
        self.load_file()
        self.get_image_list()
        self.load_audio_fea()

    def load_file(self):
        self.name_list = []
        with open(self.filename) as f:
            content = f.readlines()
        f.close()
        for i in range(len(content)):
            dirname = content[i][:-1]
            self.name_list.append(dirname)

    def load_audio_fea(self):
        """
        load audio feature
        """
        audio_feature = []
        audio_label = []

        data = h5py.File(self.audio_file)
        keys = data.keys()
        features, labels = data.get(keys[0]), data.get(keys[1])
        # numpy array to list
        convert_list = lambda x: np.ndarray.tolist(x)
        for i in range(len(features)):
            audio_feature.append(convert_list(features[i]))
            audio_label.append(convert_list(labels[i]))

        audio_label = [[int(i) for i in x] for x in audio_label]
        pairs = list(zip(audio_feature, audio_label))
        random.shuffle(pairs)
        self.audio_length = len(audio_feature)
        self.audio_feature, self.audio_label = zip(*pairs)

    def get_image_list(self):
        """
        get image list and get image from this list iterativly
        """
        if self.path == None:
            print ' no images loaded'
            return
        image_list = []
        label_list = []
        label = 0
        for i in self.name_list:
            subpath = os.path.join(self.path, i)
            for j in os.listdir(subpath):
                image_name = os.path.join(subpath, j)
                image_list.append(image_name)
                label_list.append(label)

            label = label + 1
        
        self.length = len(image_list)

        pairs = list(zip(image_list, label_list))
        random.shuffle(pairs)
        self.image_list, self.label_list = zip(*pairs)

    def next_batch(self):
        """
        generate next batch images
        return images, labels
        """
        #print ("Start location:\t%d\tEnd location:\t%d"%(self.start, self.end))
        images, audio, labels = [], [], []
        length = len(self.image_list)
        for i in range(self.start, self.end):
            label = [0] * self.class_num
            img = cv2.imread(self.image_list[i % self.length])
            img = cv2.resize(img, (40, 50))
            if len(img) == 0:
                print 'read image error:\t', self.image_list
                continue
            img = img / 255.0
            images.append(img)
            label[self.label_list[i % self.length]] = 1
            labels.append(label)
            point = random.randrange(0, self.audio_length)
            audio_fea = []
            flag = 0
            # random choose five audio matched feature
            while flag < 5:
                #print self.audio_length, point, self.label_list[i % self.length]
                if point > self.audio_length - 1:
                    point = point % self.audio_length
                if self.audio_label[point][self.label_list[i % self.length]] == 1:
                    audio_fea.extend(self.audio_feature[i % self.audio_length])
                    flag += 1
                point += 1

            audio.append(audio_fea)

        self.start = self.end
        self.end = self.end + self.batch_size

        return images, audio, labels
