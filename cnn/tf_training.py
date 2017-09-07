#encoding=utf-8
from gen_captcha import gen_captcha_text_and_image
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET

import time 
import numpy as np
import tensorflow as tf

import logging
# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS
 
text, image = gen_captcha_text_and_image()
print("verification code iamge channel:", image.shape)  # (60, 160, 3)
# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = len(text)
print("Max number of label:", MAX_CAPTCHA)   # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐
 
# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
                    
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
 
"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""
 
# 文本转向量
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')
 
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    def char2pos(c):
        if c =='_':
            k = 62
            return k
        k = ord(c)-48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map') 
        return k
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector
# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx <36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)
 
"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""
 
# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])
 
    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            #logging.debug("image,shape:%s",str(image.shape))
            if image.shape == (60, 160, 3):
                return text, image
 
    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)
 
        batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0
        batch_y[i,:] = text2vec(text)
 
    return batch_x, batch_y
 
####################################################################
def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    logging.debug("ps_hosts:%s",ps_hosts)
    logging.debug("worker_hosts:%s",worker_hosts)
    logging.debug("FLAGS:%s,%d",FLAGS.job_name,FLAGS.task_index)
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.name_scope('input'): 
            X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
            Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
        keep_prob = tf.placeholder(tf.float32) # dropout
        w_alpha = 0.01
        b_alpha = 0.1
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
            x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
            # 3 conv layer
            w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
            b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
            conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv1 = tf.nn.dropout(conv1, keep_prob)
 
            w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
            b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
            conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv2 = tf.nn.dropout(conv2, keep_prob)
 
            w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
            b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
            conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
            conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv3 = tf.nn.dropout(conv3, keep_prob)
 
            # Fully connected layer
            w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))
            b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
            dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
            dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
            dense = tf.nn.dropout(dense, keep_prob)

            with tf.name_scope('w_out'):
                w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))

            with tf.name_scope('b_out'):
                b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
            output = tf.add(tf.matmul(dense, w_out), b_out)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
            tf.summary.scalar('loss',loss) # 可视化loss常量
            # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
            predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
            max_idx_p = tf.argmax(predict, 2)
            max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
            correct_pred = tf.equal(max_idx_p, max_idx_l)

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.summary.scalar('accuracy',accuracy)

            saver = tf.train.Saver()
            global_step = tf.Variable(0)
            init_op = tf.global_variables_initializer()
            merged = tf.summary.merge_all()
            logging.debug("device done.")

        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="/home/train_logs",
                                 init_op=init_op,
                                 summary_op=None,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=300)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            logging.debug("enter supervisor")
            writer = tf.summary.FileWriter("/home/tmp/",sess.graph)
            while not sv.should_stop() :
                batch_x, batch_y = get_next_batch(64)
                _, loss_ ,step = sess.run([optimizer, loss, global_step], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
                logging.debug("step:{%d},loss:{%f}",step, loss_)

                #writer.add_summary(summary,step)
                # 每100 step计算一次准确率
                if step % 100 == 0:
                    batch_x_test, batch_y_test = get_next_batch(100)
                    summary, acc = sess.run([merged, accuracy], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                    logging.debug("step:{%d},acc:{%f}",step, acc)
                    writer.add_summary(summary,step)

                    # 如果准确率大于70%,保存模型,完成训练
                    if acc > 0.7:
                        saver.save(sess, "/home/crack_capcha.model", global_step=step)
                        break

        # Ask for all the services to stop.
        sv.stop()
        logging.info("ts_worker done.")

if __name__ == "__main__":
    LOG_FORMAT = '%(asctime)s-%(levelname)s-[%(process)d]-[%(thread)d] %(message)s (%(filename)s:%(lineno)d)'
    logging.basicConfig(format=LOG_FORMAT,level=logging.DEBUG,filename="./ts.log",filemode='w')
    tf.app.run()
