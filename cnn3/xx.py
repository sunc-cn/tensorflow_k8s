#encoding=utf-8
import os
import os.path

def classify_all_file(all_file):
    data_dict = {}
    for item in all_file:
        base_name = os.path.basename(item)
        pos = base_name.find(".jpg")
        base_chars = base_name[:pos]
        print(item)
        for c in base_chars:
            if c in data_dict.keys():
                file_list = data_dict[c]
                file_list.append(item)
                data_dict[c] = file_list
            else:
                file_list = [item]
                data_dict[c] = file_list
    print(data_dict)
    return data_dict

if __name__ == '__main__': 
    #start = time.clock()
    #LOG_FORMAT = '%(asctime)s-%(levelname)s-[%(process)d]-[%(thread)d] %(message)s (%(filename)s:%(lineno)d)'
    #logging.basicConfig(format=LOG_FORMAT,level=logging.DEBUG,filename="./ts.log",filemode='w')

    #train_crack_captcha_cnn()

    #end = time.clock()
    #print('Running time: %s Seconds'%(end - start))

    all_file = ["./abcd.jpg"]
    classify_all_file(all_file)
