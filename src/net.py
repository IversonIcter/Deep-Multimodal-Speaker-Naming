# Author: Wang Yongjie
# Email: wangyongjie@ict.ac.cn

import tensorflow as tf
import numpy as np
import time
from data import preprocess

class Network(object):
    """
    create network for face recognition
    """
    def __init__(self, batch_size, class_num):
        """
        init class parameter
        -batch_size:         train batch
        -class_num:          the last fc layers length 
        """
        self.batch_size = batch_size
        self.class_num = class_num

    def dropout(self, x, keep_prob):
        """
        dropout function
        """
        return tf.nn.dropout(x, keep_prob)

    def avg_pooling(self, x, filter_height, filter_width, stride_y, stride_x, padding,  name):
        """
        -x:             input tensor
        -filter_height: pooling size(height)
        -filter_width:  pooling size(width)
        -stride_y:      pooling stride (y aix)
        -stride_x:      pooling stride (x aix)
        -padding:       padding option, default padding is SAME
        -name:          layer name of pooling

        """
        return tf.nn.avg_pool(x, ksize = [1, filter_height, filter_width, 1], 
                strides = [1, stride_y, stride_x, 1],
                padding = padding,
                name = name)

    def conv(self, x, filter_height, filter_width, num_filter, stride_y, stride_x, name, padding = 'VALID'):
        """
        -x:                 input tensor [ batch_size, height, width, channels]
        -filter_height:     filter height [height, width, in_channels, out_channels]
        -filter_width:      filter width
        -num_filter:        kernel numbers
        -stride_y:          height_directory stride [1, stride_y, stride_x, 1]
        -stride_x:          width_directory stride
        -padding:           padding type, default is "VALID"
        -name:              convolution name
        """
        input_channels = int(x.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable("weights", shape = [filter_height, filter_width, input_channels, num_filter], initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-4))
            biases = tf.get_variable("biases", shape = [num_filter],  initializer = tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(x, weights, [1, stride_y, stride_x, 1], padding)
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias, name = scope.name)
            return relu

    def fc(self, x, num_in, num_out, name, relu = True):
        """
        -x:        input tensor
        -num_in:   input length
        -num_out:  output length
        -relu:     default activated function relu
        """
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable("weights", shape = [num_in, num_out],initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-4))
            biases = tf.get_variable("biases", shape = [num_out], initializer = tf.constant_initializer(0.1))
            fc = tf.add(tf.matmul(x, weights), biases, name = scope.name)
            #result = tf.nn.xw_plus_b(x, weights, biases, name = scope.name)

            if relu:
                relu = tf.nn.relu(fc)
                return relu
            else:
                return fc

    def create_network(self, x, keep_prob):
        assert(x != None)
        conv1 = self.conv(x, 15, 15, 48, 1, 1, name = 'conv1', padding = "VALID")
        pool1 = self.avg_pooling(conv1, 2, 2, 2, 2, padding = "SAME", name = 'pool1')
        conv2 = self.conv(pool1, 5, 4, 256, 1, 1, name = 'conv2', padding = "VALID")
        pool2 = self.avg_pooling(conv2, 2, 2, 2, 2, padding = "SAME", name = 'pool2')
        conv3 = self.conv(pool2, 7, 5, 1024, 1, 1, name = "conv3", padding = "VALID")

        flatten = tf.reshape(conv3, [-1, 1024])
        fc1 = self.fc(flatten, 1024, 1024, name = "fc1")
        dp1 = self.dropout(fc1, keep_prob)
        fc2 = self.fc(dp1, 1024, 2048, name = "fc2")
        dp2 = self.dropout(fc2, keep_prob)
        fc3 = self.fc(dp2, 2048, self.class_num, name = "fc3", relu = False)
        return fc1, fc3

    def train_network(self, sess, data_dir, epoch,  save_name, test_dir):
        """
        session:        tensorflow session
        data:           data directory
        save_name:      tensorflow ckpt name
        """

        Process = preprocess(data_dir, self.batch_size, self.class_num)
        Test = preprocess(test_dir, 20000, 6)
        iteration = Process.length / self.batch_size
        x_in = tf.placeholder(tf.float32, [None, 50, 40, 3])
        y_in = tf.placeholder(tf.float32, [None, self.class_num])
        keep_prob = tf.placeholder(tf.float32)


        f = open("acc-face.txt", "w")
        f.write("train acc\n")


        fc1, predict = self.create_network(x_in, keep_prob)
        static = tf.equal(tf.argmax(predict, 1), tf.argmax(y_in, 1))
        accuracy = tf.reduce_mean(tf.cast(static, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_in, logits = predict)
        loss = tf.reduce_mean(loss)
        optim = tf.train.AdamOptimizer(0.0001).minimize(loss)

        tf.summary.scalar("loss", loss)
        saver = tf.train.Saver(max_to_keep = 10)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/", sess.graph)
        sess.run(tf.global_variables_initializer())


        for i in range(epoch):
            for j in range(iteration):
                input_x, label_x = Process.next_batch()
                #print label_x
                #print("%s\tEpoch\t%d\tIteration\t%d"%(time.asctime(), i, j))
                #conv1, pool1, conv2, pool2, conv3, _predict = sess.run([self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, predict], feed_dict = {x_in: input_x, y_in: label_x})
                #print _predict
                sess.run(optim, feed_dict = {x_in: input_x, y_in:label_x, keep_prob:0.5})

                if j % 5 == 0:
                    #_predict = sess.run(predict, feed_dict = {x_in: input_x, y_in:label_x, keep_prob:0.5})
                    #print _predict#, label_x
                    _acc, _loss = sess.run([accuracy, loss], feed_dict = {x_in: input_x, y_in:label_x, keep_prob:0.5})
                    print("%s\tEpoch\t%d\tIteration\t%d\tAccuracy\t%f\tLoss\t%f"%(time.asctime(), i, j, _acc.item(), _loss.item()))
                    results = sess.run(merged, feed_dict={x_in: input_x, y_in:label_x, keep_prob:0.5})
                    writer.add_summary(results, i * iteration + j)


            if (i + 1) % 2 == 0:
                save_path = saver.save(sess, save_name, global_step = (i+i))
                test_x, test_y = Test.next_batch()
                _acc = sess.run(accuracy, feed_dict = {x_in: test_x, y_in:test_y, keep_prob:1})
                f.write(str(i+1) + "\t" + str(_acc) + '\n')
                print("Test Accuracy\t%f"%_acc)

        f.close()

if __name__ == "__main__":
    trainset = "/home/wyj/multi-modal/data/Friends/"
    testset = "/home/wyj/multi-modal/data/test/"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    FaceNet = Network(64, 6)
    FaceNet.train_network(sess, trainset, 30, "./model/face-model.ckpt", testset) 
