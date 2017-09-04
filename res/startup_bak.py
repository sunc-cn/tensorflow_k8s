#!/usr/local/bin/python3.6
#encoding=utf-8

import logging
import subprocess
import json
import os.path

def call_shell(command_str):
    obj = subprocess.Popen(
            command_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True)
    return (obj.stdout.read().decode("utf-8"),obj.stderr.read().decode("utf-8"))

def check_k8s_err(err_out_str):
    #print(err_out_str)
    err_str = err_out_str.replace("\n","") 
    if 0 == len(err_str):
        return False
    return True

def k8s_create_ts_image(ts_id_str):
    command_str = "kubectl run " + ts_id_str + " --image=tensorflow/tensorflow"
    (s,e) = call_shell(command_str)
    if check_k8s_err(e):
        print("k8s_create_ts_image,error:",e)
        return False
    return True

def k8s_check_ts_image(ts_id_str):
    command_str = "kubectl get po -o json"
    (s,e) = call_shell(command_str)
    if check_k8s_err(e):
        print("k8s_check_ts_image,error:",e)
        return False
    json_obj = json.loads(s)
    items = json_obj["items"]
    image_count = 0
    for i in items:
        meta = i["metadata"]
        spec = i["spec"]
        image_name = meta["name"]
        #print(image_name)
        if image_name.find(ts_id_str) != -1:
            image_count += 1
    if image_count == 0:
        return False
    return True
     
def k8s_scale_ts_image(ts_id_str,count):
    command_str = "kubectl scale --replicas= " + str(count) + " deploy/" +  ts_id_str
    (s,e) = call_shell(command_str)
    if check_k8s_err(e):
        print("k8s_scale_ts_image,error:",e)
        return False
    return True

def k8s_get_ts_image(ts_id_str):
    command_str = "kubectl get po -o wide -o json"
    (s,e) = call_shell(command_str)
    if check_k8s_err(e):
        print("k8s_get_ts_image,error:",e)
        return {}
    json_obj = json.loads(s)
    items = json_obj["items"]
    image_dict = {}
    for i in items:
        meta = i["metadata"]
        spec = i["spec"]
        status = i["status"]
        image_name = meta["name"]
        #print(image_name)
        if image_name.find(ts_id_str) != -1:
            pod_ip = status["podIP"]
            # and check if its Runing
            running = status["phase"]
            if running == "Running":
                image_dict[image_name] = pod_ip
    #print(image_dict)
    return image_dict

def k8s_deploy_file_only(file_path,dst_image):
    base_name = os.path.basename(file_path)
    command_str = "kubectl cp " + file_path + " " + dst_image + ":/home/" + base_name
    print(command_str)
    (s,e) = call_shell(command_str)
    if check_k8s_err(e):
        print("k8s_deploy_file_only,error:",e)
        return False
    return True

def k8s_distribute_ts_images(ts_id_str,ps_count,worker_count):
    all_count = ps_count + worker_count
    images = k8s_get_ts_image(ts_id_str)
    if len(images) < all_count:
        logging.warning("error,quit,request resources is not enough,request:%d,actual:%d"%(all_count,len(images)))
        return
    elif len(images) > all_count:
        logging.warning("warning,do nothing,request resources is less than images,request:%d,actual:%d"%(all_count,len(images)))
    # distribute ps images
    images_list = []
    for (k,v) in images.items():
        images_list.append((k,v))
    ps_list = []
    if ps_count > 0:
        ps_list = images_list[0:ps_count]
        images_list = images_list[ps_count:]
    logging.debug("ps_list:%s",ps_list)
    # distribute worker images
    worker_list = []
    if worker_count > 0:
        worker_list = images_list[0:worker_count]
        images_list = images_list[worker_count:]
    logging.debug("worker_list:%s",worker_list)
    logging.debug("left_list:%s",images_list)
    return (ps_list,worker_list)
   
def k8s_startup_ts(ts_id_str,ps_hosts,worker_hosts,job_name,task_index):
    command_str = "kubectl exec -it "+ts_id_str 
    command_str += " -- python /home/startup_worker.py --ps_hosts=\""+ ps_hosts + "\""
    command_str += " --worker_hosts=\"" + worker_hosts + "\""
    command_str += " --job_name=" + job_name + " --task_index=" + str(task_index)
    #print(command_str)
    if not k8s_check_worker_is_running(ts_id_str):
        print(command_str)
        (s,e) = call_shell(command_str)
        if check_k8s_err(e):
            print("k8s_startup_ts,error:%s,command_str:%s",e,command_str)
        else:
            print("k8s_startup_ts,s:%s,e:%s",s,e)
    pass

def k8s_check_worker_is_running(ts_id_str):
    command_str = "kubectl exec -it " + ts_id_str + " -- ps -ef | grep \"/home/ts_worker.py\""
    (s,e) = call_shell(command_str)
    if len(s) == 0:
        return False
    return True
    

def k8s_deploy_ts(ts_id_str,ps_count,worker_count,ts_worker_file):
    (ps_list,worker_list) = k8s_distribute_ts_images(ts_id_str,ps_count,worker_count)
    ps_port = 20000
    worker_port = 20001
    ps_ips_str = ""
    for item in ps_list:
        (image,ip) = item
        ps_ips_str += ip+":"+str(ps_port) + ","
        ret = k8s_deploy_file_only(ts_worker_file,image)
        #print("deploy ps ret:",ret)
        ret = k8s_deploy_file_only("./startup_worker.py",image)
    worker_ips_str = ""
    for item in worker_list:
        (image,ip) = item
        worker_ips_str += ip+":"+str(worker_port) + ","
        ret = k8s_deploy_file_only(ts_worker_file,image)
        #print("deploy worker ret:",ret)
        ret = k8s_deploy_file_only("./startup_worker.py",image)
    if len(ps_ips_str) > 1:
        ps_ips_str = ps_ips_str[:len(ps_ips_str)-1]
    if len(worker_ips_str) > 1:
        worker_ips_str = worker_ips_str[:len(worker_ips_str)-1]
    #print(ps_ips_str)
    #print(worker_ips_str)
    logging.debug("finish copy files")
    ps_index = 0
    for item in ps_list:
        (image,ip) = item
        k8s_startup_ts(image,ps_ips_str,worker_ips_str,"ps",ps_index)
        ps_index += 1
    logging.debug("finish startup ps server")
    worker_index = 0
    for item in worker_list:
        (image,ip) = item
        k8s_startup_ts(image,ps_ips_str,worker_ips_str,"worker",worker_index)
        worker_index += 1
    logging.debug("finish startup worker server")
    pass 

def k8s_kill_all_ts_process(ts_id_str):
    image_dict = k8s_get_ts_image(ts_id_str)
    for (k,v) in image_dict.items():
        k8s_kill_single_ts_process(k)
def k8s_kill_single_ts_process(contianer_name):
    command_str = "kubectl exec -it "+ contianer_name+" -- ps -ef |grep \"/home/ts_worker.py\" | awk -F \" \" '{print $2}'"
    (s,e) = call_shell(command_str)
    if len(s)!=0:
        command_str = "kubectl exec -it "+ contianer_name+ " -- kill -9 " + s
        print(command_str)
        (s,e) = call_shell(command_str)
    pass

if __name__ == "__main__":
    LOG_FORMAT = '%(asctime)s-%(levelname)s-[%(process)d]-[%(thread)d] %(message)s (%(filename)s:%(lineno)d)'
    #logging.basicConfig(format=LOG_FORMAT,level=logging.DEBUG,filename="./ts.log",filemode='w')
    logging.basicConfig(format=LOG_FORMAT,level=logging.DEBUG)

    #logging.debug("hello")
    k8s_deploy_ts("ts-0002",1,4,"./ts_worker.py")
    #k8s_kill_all_ts_process("ts-0002")
    pass
