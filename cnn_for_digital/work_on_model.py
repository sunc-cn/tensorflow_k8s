#encoding=utf-8
from training import *
import logging
import time
import numpy as np
import tensorflow as tf
 
class DigitalOCR():
    def __init__(self):
        self.output = crack_captcha_cnn()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.saver.restore(sess, tf.train.latest_checkpoint('./'))
        self.predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        pass
    def recognize(self,im):
        image = np.array(im)
        image = convert2gray(image) #生成一张新图
        image = image.flatten() / 255 # 将图片一维化
        text_list = self.sess.run(self.predict, feed_dict={X: [image], keep_prob: 1})
        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i*CHAR_SET_LEN + n] = 1
            i += 1
        predict_text = vec2text(vector)
        return predict_text
        pass
    def clean():
        if self.sess != None:
            self.sess.close()
