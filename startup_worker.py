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
    command_str = "python " + dir_name + "/ts_worker.py"
    command_slice = ["nohup","python",dir_name+"/ts_worker.py"]
    for i in args:
        command_str += " " + i
        command_slice.append(i)
    command_str += " &"
    command_slice.append("&")
    print(command_str)
    #os.popen(command_str) 
    devnull = open(os.devnull,'wb')
    p = subprocess.Popen(command_slice,stdout=devnull,stderr=devnull)
    print(p.pid)
    pass
