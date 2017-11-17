#encoding=utf-8
from training import *
import logging
import time
import numpy as np
import tensorflow as tf
 
def recognize_img(img_path):
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        base_name = os.path.basename(img_path)
        captcha_image = Image.open(img_path)
        image = np.array(captcha_image)
        image = convert2gray(image) #生成一张新图
        image = image.flatten() / 255 # 将图片一维化
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i*CHAR_SET_LEN + n] = 1
            i += 1
        predict_text = vec2text(vector)
        print(vector)
        print("原始文件: {}  预测: {}".format(img_path, predict_text))
        

if __name__ == '__main__': 
    start = time.clock()
    LOG_FORMAT = '%(asctime)s-%(levelname)s-[%(process)d]-[%(thread)d] %(message)s (%(filename)s:%(lineno)d)'
    logging.basicConfig(format=LOG_FORMAT,level=logging.DEBUG,filename="./ts.log",filemode='w')
    
    end = time.clock()
    print('Running time: %s Seconds'%(end - start))
