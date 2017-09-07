#!/usr/bin/env python
#encoding=utf-8

import os
import sys
import os.path
import time
import subprocess

if __name__ == "__main__":
    dir_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(dir_path)
    args = sys.argv[1:] 
    #command_slice = ["nohup","python",dir_name+"/ts_worker.py"]
    command_slice = ["nohup","python",dir_name+"/tf_training.py"]
    for i in args:
        command_slice.append(i)
    command_slice.append("&")
    print(" ".join(command_slice))
    devnull = open(os.devnull,'wb')
    p = subprocess.Popen(command_slice,stdout=devnull,stderr=devnull)
    print(p.pid)
    pass
