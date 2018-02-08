# 2018.01.26 17:02:01 +08
# Author: Wang Yongjie
# Email: wangyongjie@ict.ac.cn
import cv2
import sys
import os
import random

class preprocess(object):
    """
    generate image batches
    """

    def __init__(self, path, batch_size, class_num):
        """
        init class parameter
        path:               image directory
        batch_size:         batch_size
        """
        self.path = path
        self.batch_size = batch_size
        self.class_num = class_num
        self.start = 0
        self.end = batch_size
        self.get_image_list()

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
        for i in os.listdir(self.path):
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
        images, labels = [], []
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

        self.start = self.end
        self.end = self.end + self.batch_size

        return images, labels
