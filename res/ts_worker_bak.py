#!/usr/bin/env python
#encoding=utf-8

import logging
import logging.handlers
import tensorflow as tf
import numpy as np

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS



def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    logging.debug("ps_hosts:%s",ps_hosts)
    logging.debug("worker_hosts:%s",worker_hosts)
    # clear illegal data
    cp_ps_hosts = ps_hosts
    ps_hosts = []
    cp_worker_hosts = worker_hosts
    worker_hosts = []
    for i in cp_ps_hosts:
        if len(i) != 0:
            ps_hosts.append(i)
    for i in cp_worker_hosts:
        if len(i) != 0:
            worker_hosts.append(i)

    # Create a cluster from the parameter server and worker hosts.
    if len(ps_hosts) != 0 and len(worker_hosts) != 0:
        cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    elif len(ps_hosts) != 0:
        cluster = tf.train.ClusterSpec({"ps": ps_hosts})
    elif len(worker_hosts) != 0:
        cluster = tf.train.ClusterSpec({"worker": worker_hosts})
    else:
        logging.warning("no ps or woker set,quit.")
        return

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    server.join()


if __name__ == "__main__":
    LOG_FORMAT = '%(asctime)s-%(levelname)s-[%(process)d]-[%(thread)d] %(message)s (%(filename)s:%(lineno)d)'
    logging.basicConfig(format=LOG_FORMAT,level=logging.DEBUG)
    tf.app.run()
