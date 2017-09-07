#!/usr/local/bin/python3.6
#encoding=utf-8

import sys
import argparse
import logging
import subprocess
import json
import os.path
import time

# call_shell, execute a shell command, return (stdout_str,stderr_str)
def call_shell(command_str):
    obj = subprocess.Popen(
            command_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True)
    return (obj.stdout.read().decode("utf-8"),obj.stderr.read().decode("utf-8"))

# k8s_check_err,check if occur error.
def k8s_check_err(err_out_str):
    #print(err_out_str)
    err_str = err_out_str.replace("\n","") 
    if 0 == len(err_str):
        return False
    return True

# call_k8s_command, call a k8s command,ruturn true if no error occur.
def call_k8s_command(command_str):
    (s,e) = call_shell(command_str)
    if k8s_check_err(e):
        logging.error("call_k8s_command,do command_str:%s,error:%s",command_str,e)
        return False
    return True

# k8s_destory_deployment,destory a deployment from k8s's cluster.
def k8s_destory_deployment(deployment_name):
    # example: kubectl delete deployment ts-0002
    command_str = "kubectl delete deployment "+ deployment_name 
    return call_k8s_command(command_str)

# k8s_create_tf_deployment,create new container from tensorflow image.
def k8s_create_tf_deployment(deployment_name):
    # TODO, use your tensorflow's image
    #command_str = "kubectl run " + deployment_name+ " --image=tensorflow/tensorflow"
    command_str = "kubectl run " + deployment_name+ " --image=ccr.ccs.tencentyun.com/g7_docker/tensorflow"
    return call_k8s_command(command_str)

# k8s_check_tf_deployment,check if the specify deployment_name deploy on k8s success.
def k8s_check_tf_deployment(deployment_name):
    command_str = "kubectl get po -o json"
    (s,e) = call_shell(command_str)
    if k8s_check_err(e):
        print("k8s_check_tf_deployment,error:",e)
        return False
    json_obj = json.loads(s)
    items = json_obj["items"]
    image_count = 0
    for i in items:
        meta = i["metadata"]
        spec = i["spec"]
        image_name = meta["name"]
        #print(image_name)
        if image_name.find(deployment_name) != -1:
            image_count += 1
    if image_count == 0:
        return False
    return True
     
# k8s_scale_tf_deployment,scale the contianer's count of the specify deployment_name's deployment to contianer_count.
def k8s_scale_tf_deployment(deployment_name,contianer_count):
    command_str = "kubectl scale --replicas=" + str(contianer_count) + " deploy/" + deployment_name 
    return call_k8s_command(command_str)

# k8s_get_tf_containers,get the deployment_name's containers
def k8s_get_tf_containers(deployment_name):
    command_str = "kubectl get po -o wide -o json"
    (s,e) = call_shell(command_str)
    if k8s_check_err(e):
        print("k8s_get_tf_containers,error:",e)
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
        if image_name.find(deployment_name) != -1:
            # and check if its Runing
            running = status["phase"]
            if running == "Running":
                pod_ip = status["podIP"]
                image_dict[image_name] = pod_ip
    #print(image_dict)
    return image_dict

# k8s_deploy_tf_file,cp the file_path's file to spec container's /home/
def k8s_deploy_tf_file(file_path,contianer_name):
    base_name = os.path.basename(file_path)
    command_str = "kubectl cp " + file_path + " " + contianer_name+ ":/home/" + base_name
    print(command_str)
    return call_k8s_command(command_str)

# k8s_distribute_tf_containers,distribute ps's containers and worker's containers
def k8s_distribute_tf_containers(deployment_name,ps_count,worker_count):
    all_count = ps_count + worker_count
    images = k8s_get_tf_containers(deployment_name)
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

# k8s_startup_tf_process,startup a tf process
def k8s_startup_tf_process(deployment_name,ps_hosts,worker_hosts,job_name,task_index,tf_file):
    command_str = "kubectl exec -it "+deployment_name 
    command_str += " -- python /home/startup_worker.py --ps_hosts=\""+ ps_hosts + "\""
    command_str += " --worker_hosts=\"" + worker_hosts + "\""
    command_str += " --job_name=" + job_name + " --task_index=" + str(task_index)
    #print(command_str)
    if not k8s_check_worker_is_running(deployment_name,tf_file):
        print(command_str)
        (s,e) = call_shell(command_str)
        if k8s_check_err(e):
            logging.error("k8s_startup_tf_process,error:%s,command_str:%s",e,command_str)
        else:
            logging.debug("k8s_startup_tf_process,s:%s,e:%s",s,e)
    pass

# k8s_check_worker_is_running,check if the tf's process is running
def k8s_check_worker_is_running(deployment_name,tf_file):
    base_name = os.path.basename(tf_file)
    command_str = "kubectl exec -it " + deployment_name + " -- ps -ef | grep \"/home/" + base_name + "\""
    (s,e) = call_shell(command_str)
    if len(s) == 0:
        return False
    return True

#  k8s_deploy_tf, copy files to tf's container and startup a tf process
def k8s_deploy_tf(deployment_name,ps_count,worker_count,tf_worker_file):
    (ps_list,worker_list) = k8s_distribute_tf_containers(deployment_name,ps_count,worker_count)
    ps_port = 20000
    worker_port = 20001
    ps_ips_str = ""
    for item in ps_list:
        (image,ip) = item
        ps_ips_str += ip+":"+str(ps_port) + ","
        ret = k8s_deploy_tf_file(tf_worker_file,image)
        #print("deploy ps ret:",ret)
        ret = k8s_deploy_tf_file("./startup_worker.py",image)
    worker_ips_str = ""
    for item in worker_list:
        (image,ip) = item
        worker_ips_str += ip+":"+str(worker_port) + ","
        ret = k8s_deploy_tf_file(tf_worker_file,image)
        #print("deploy worker ret:",ret)
        ret = k8s_deploy_tf_file("./startup_worker.py",image)
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
        k8s_startup_tf_process(image,ps_ips_str,worker_ips_str,"ps",ps_index,tf_worker_file)
        ps_index += 1
    logging.debug("finish startup ps server")
    worker_index = 0
    for item in worker_list:
        (image,ip) = item
        k8s_startup_tf_process(image,ps_ips_str,worker_ips_str,"worker",worker_index,tf_worker_file)
        worker_index += 1
    logging.debug("finish startup worker server")
    pass 

# k8s_kill_tf_processes,kill all tf's processes
def k8s_kill_tf_processes(deployment_name):
    image_dict = k8s_get_tf_containers(deployment_name)
    for (k,v) in image_dict.items():
        k8s_kill_single_tf_process(k)

# k8s_kill_single_tf_process,kill a single tf process
def k8s_kill_single_tf_process(contianer_name):
    command_str = "kubectl exec -it "+ contianer_name+" -- ps -ef |grep py |grep \"/home/\" | awk -F \" \" '{print $2}'"
    (s,e) = call_shell(command_str)
    if len(s)!=0:
        command_str = "kubectl exec -it "+ contianer_name+ " -- kill -9 " + s
        print(command_str)
        (s,e) = call_shell(command_str)
    pass

#  k8s_copy_file_to_tf, copy files to tf's container
def k8s_copy_file_to_tf(deployment_name,file_path):
    logging.debug("k8s_copy_file_to_tf start,deployment_name:%s,file_path:%s",deployment_name,file_path)
    images = k8s_get_tf_containers(deployment_name)
    for k,v in images.items():
        k8s_deploy_tf_file(file_path,k)
    logging.debug("k8s_copy_file_to_tf end,deployment_name:%s,file_path:%s",deployment_name,file_path)

def k8s_exec_command_on_all_tfs(deployment_name,command_str):
    logging.debug("k8s_exec_command_on_all_tfs,start,deployment_name:%s,command_str:%s",deployment_name,deployment_name)
    images = k8s_get_tf_containers(deployment_name)
    for k,v in images.items():
        tmp_command_str = "kubectl exec -it " + k + " -- " + command_str
        print(tmp_command_str)
        call_k8s_command(tmp_command_str)
    logging.debug("k8s_exec_command_on_all_tfs,end,deployment_name:%s,command_str:%s",deployment_name,deployment_name)

def main(args):
    if args.destory:
        logging.debug("k8s_destory_deployment:%s,start",args.destory)
        k8s_destory_deployment(args.destory)
        logging.debug("k8s_destory_deployment:%s,finished",args.destory)
    if args.init and not args.n:
        logging.debug("k8s_create_tf_deployment:%s,start",args.init)
        k8s_create_tf_deployment(args.init)
        logging.debug("k8s_create_tf_deployment:%s,finished",args.init)
    elif args.init and args.n:
        logging.debug("k8s_create_tf_deployment:%s,count:%s,start",args.init,args.n)
        k8s_create_tf_deployment(args.init)
        k8s_scale_tf_deployment(args.init,int(args.n))
        wait_times = 2*int(args.n)
        while wait_times > 0:
            time.sleep(1)
            dst = k8s_get_tf_containers(args.init) 
            if len(dst) == args.n:
                break
            wait_times -= 1
        logging.debug("k8s_create_tf_deployment:%s,count:%s,finished",args.init,args.n)
    if args.startup and args.tf and 0 != (int(args.ps) + int(args.wk)):
        logging.debug("k8s_deploy_tf:%s,ps:%s,worker:%s,tf file:%s,start",
                args.startup,args.ps,args.wk,args.tf)
        ps = 0
        wk = 0
        if args.ps == "":
            ps = 0
        else:
            ps = int(args.ps)
        if args.wk == "":
            wk = 0
        else:
            wk = int(args.wk)
        k8s_deploy_tf(args.startup,ps,wk,args.tf)
        logging.debug("k8s_deploy_tf:%s,ps:%s,worker:%s,tf file:%s,finished",
                args.startup,args.ps,args.wk,args.tf)

    if args.shutdown:
        logging.debug("k8s_kill_tf_processes:%s,start",args.shutdown)
        k8s_kill_tf_processes(args.shutdown)
        logging.debug("k8s_kill_tf_processes:%s,finished",args.shutdown)
    if args.cp and args.f:
        logging.debug("k8s_copy_file_to_tf,start,cp:%s,file:%s",args.cp,args.f)
        k8s_copy_file_to_tf(args.cp,args.f)
        logging.debug("k8s_copy_file_to_tf,end,cp:%s,file:%s",args.cp,args.f)
    if args.run and args.c:
        logging.debug("k8s_exec_command_on_all_tfs,run:%s,command_str:%s,start",args.run,args.c)
        k8s_exec_command_on_all_tfs(args.run,args.c)
        logging.debug("k8s_exec_command_on_all_tfs,run:%s,command_str:%s,end",args.run,args.c)
        pass
        

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("please with -h option get help information")
        exit
    LOG_FORMAT = '%(asctime)s-%(levelname)s-[%(process)d]-[%(thread)d] %(message)s (%(filename)s:%(lineno)d)'
    #logging.basicConfig(format=LOG_FORMAT,level=logging.DEBUG,filename="./ts.log",filemode='w')
    logging.basicConfig(format=LOG_FORMAT,level=logging.DEBUG)
    parser = argparse.ArgumentParser("a kubenetes commands wrapper for tensorflow.")
    parser.add_argument("-destory",help="<-destory=deployment_name>, destory a series of contianer with specify name.example: ./startup.py -destory ts-0002")
    parser.add_argument("-init",help="<-init=deployment_name>, init a series of contianer with specify name,default without --n init a contianer.example: ./startup.py -init ts-0002 --n 5")
    parser.add_argument("--n",help="<--n=number>, set the number of contianers")
    parser.add_argument("-startup",help="<-startup=deployment_name>, startup tensorflow processes,example: ./startup.py -startup ts-0002 --ps 1 --wk 4 --tf ./ts_worker.py")
    parser.add_argument("--ps",help="<--ps=number>, set the number of ps worker")
    parser.add_argument("--wk",help="<--wk=number>, set the number of computation worker")
    parser.add_argument("--tf",help="<--tf=file_path>, set the file of tensorflow application")
    parser.add_argument("-shutdown",help="<-shutdown=deployment_name>, shutdown tensorflow processes,with parameter,deployment name.example: ./startup.py -shutdown ts-0002")
    parser.add_argument("-cp",help="<-cp=deployment_name>, copy file to all container.example: ./startup.py -cp ts-0002 --f ./ts_worker.py")
    parser.add_argument("--f",help="<-f=file_path>, set the file path be copied.")
    parser.add_argument("-run",help="<-run=deployment_name>, exec command on all container.example: ./startup.py -run ts-0002 --c ls")
    parser.add_argument("--c",help="<-c=command_str>, set the command be executed.")

    args = parser.parse_args()
    main(args)
    pass
