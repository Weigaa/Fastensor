import torch
import os
import sys
import numpy as np
import kvikio
import cupy
import time
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
# from nvidia.dali import pipeline_def, fn
# import nvidia.dali.plugin.pytorch
import csv
import copy
import ast

#创建一个读写字典，保存不同数据块大小下的不同API读写速度
writedic = {}
readdic = {}
wrdic = {}
mydic = {}
#创建一个全局刷新变量
flag = 0
#定义变量确定总共的方法个数(包含了方法和选择好的最优方法)
totalmethods = 4
totalsamples = 80
# @pipeline_def(batch_size=1, num_threads=4, device_id=0)
# def pipe_gds(filename):
#     data = fn.readers.numpy(device='gpu', file_root='.', files=filename, register_buffers = True, cache_header_information=True)
#     return data

def save(Inputtensor,path='myfile',flush = False, type = "w", use_dic = "None" ,policy = "None"):
    global writedic,readdic,flag, BSZ, wrdic, mydic
    #size = sys.getsizeof(Inputtensor)
    size=1
    print("size is", size)
    tensor_shape = 0
    tensor_type = 0
    Inputfile = []

    #判断是否使用给定的字典：
    if use_dic != "None" and not bool(mydic):
        path = use_dic
        with open(path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                mydic[round(float(row[0]),6)] = ast.literal_eval(row[1])

    if mydic:#没跑
        #保留6位小数以保证size对齐
        size = round(size, 6)
        # print("size is ", size)
        # print("mydic is", mydic)
        bestmethod = mydic[size][-1]
        if  bestmethod == "cpy":
            # print("save as cupy")
            begin = time.time()
            # Inputcupy = cupy.asarray(Inputtensor)
            if tensor_type is not torch.bool:
                Inputcupy = cupy.array(np.asarray(Inputtensor), order = 'C')
            else:
                Inputcupy = cupy.asarray(Inputtensor.cpu())
            path = path + ".cpy"
            f = kvikio.CuFile(path, "w")
            f.write(Inputcupy)
            f.close()
            end = time.time()
            duration = end - begin
            # print("cupy save size ",size," bandwidth is", size/duration)
            Inputfile = [path, tensor_shape, tensor_type, size]
        if  bestmethod == "npy":
            # print("save as Numpy")
            Inputnumpy = Inputtensor.numpy()
            path = path + ".npy"
            np.save(path, Inputnumpy)
            Inputfile = [path, tensor_shape, tensor_type, size]
        if  bestmethod == "pt":
            # print("save as Torch")
            path = path + ".pt"
            torch.save(Inputtensor, path)
            Inputfile = [path, tensor_shape, tensor_type, size]
        return Inputfile

    if policy != "None":
        if policy == "cpy":
            # print("save as cupy")
            begin = time.time()
            tensor_info = []
            total_info = []
            #cupy暂时不能处理bool和带有梯度的GPU张量
            if tensor_type is not torch.bool:
                outputlist = list(Inputtensor.values())
                outfile = outputlist[0].view(-1)  #展开 
                tensor_info.append([outputlist[0].shape, outputlist[0].dtype, len(outputlist[0].view(-1))])
                for i in outputlist[1:]:
                    outfile = torch.cat((outfile,i.view(-1))) #拼接成一维数组
                    tensor_info.append([i.shape, i.dtype, len(i.view(-1))])
                Inputcupy = cupy.asarray(outfile, order='C')
                total_info = [outfile.shape,outfile.dtype]
            else:
                Inputcupy = cupy.asarray(Inputtensor.cpu())
            path = path + ".cpy"
            f = kvikio.CuFile(path, "w")
            f.write(Inputcupy)
            f.close()
            #保存张量字典的名字和对应的张量类型信息
            np.save(path + "keys.npy", list(Inputtensor.keys()))
            np.save(path + "tensorinfo.npy", tensor_info)
            np.save(path + "totalinfo.npy", total_info)
            end = time.time()
            duration = end -begin
            # print("cupy save size ",size," bandwidth is", size/duration)
            Inputfile = [path, tensor_shape, tensor_type, size]
        if policy  == "npy":
            # print("save as Numpy")
            begin = time.time()
            Inputnumpy = np.array(Inputtensor)
            path = path + ".npy"
            np.save(path, Inputnumpy)
            end = time.time()
            duration = end -begin
            # print("numpy save size ",size," bandwidth is", size/duration)
            Inputfile = [path, tensor_shape, tensor_type, size]
        if policy  == "pt":
            # print("save as Torch")
            begin = time.time()
            path = path + ".pt"
            torch.save(Inputtensor, path)
            end = time.time()
            duration = end - begin
            # print("numpy save size ",size," bandwidth is", size/duration)
            Inputfile = [path, tensor_shape, tensor_type, size]
        if policy  == "dali":
            # print("save as DALI")
            begin = time.time()
            Inputnumpy = Inputtensor.cpu().numpy()
            path = path + ".dali.npy"
            np.save(path, Inputnumpy)
            end = time.time()
            duration = end - begin
            # print("dali save size ",size," bandwidth is", size/duration)
            Inputfile = [path, tensor_shape, tensor_type, size]
        return Inputfile

    if flush or flag == 3:
        # print("readdic is",readdic)
        # print("writedic is", writedic)
        writedic = {}
        readdic = {}
        wrdic = {}
        # flag = 0
    # if flag == 2000:
    #     # save on CSV file
    #     readname = "readdic.csv"
    #     writename = "writedic.csv"
    #     wrname = "wrdic.csv"
    #     #保存文件
    #     with open(readname, "a", newline='') as csv_file:
    #         writer = csv.writer(csv_file)
    #         for key, value in readdic.items():
    #             writer.writerow([key, value])
    #     with open(writename, "a", newline='') as csv_file2:
    #         writer = csv.writer(csv_file2)
    #         for key, value in writedic.items():
    #             writer.writerow([key, value])
    #     with open(wrname, "a", newline='') as csv_file3:
    #         writer = csv.writer(csv_file3)
    #         for key, value in wrdic.items():
    #             writer.writerow([key, value])
    # flag += 1

    #判断是否为读写模式("W+r")
    if type == "w+r" and size in wrdic:
        # print("进入 w+r模式 ",wrdic[size][-1])
        bestmethod = wrdic[size][-1]
        if  bestmethod == "cpy":
            # print("save as cupy")
            begin = time.time()
            tensor_info = []
            total_info = []
            #cupy暂时不能处理bool和带有梯度的GPU张量
            if tensor_type is not torch.bool:
                outputlist = list(Inputtensor.values())
                outfile = outputlist[0].view(-1)  #展开 
                tensor_info.append([outputlist[0].shape, outputlist[0].dtype, len(outputlist[0].view(-1))])
                for i in outputlist[1:]:
                    outfile = torch.cat((outfile,i.view(-1))) #拼接成一维数组
                    tensor_info.append([i.shape, i.dtype, len(i.view(-1))])
                Inputcupy = cupy.asarray(outfile, order='C')
                total_info = [outfile.shape,outfile.dtype]
            else:
                Inputcupy = cupy.asarray(Inputtensor.cpu())
            path = path + ".cpy"
            f = kvikio.CuFile(path, "w")
            f.write(Inputcupy)
            f.close()
            #保存张量字典的名字和对应的张量类型信息
            np.save(path + "keys.npy", list(Inputtensor.keys()))
            np.save(path + "tensorinfo.npy", tensor_info)
            np.save(path + "totalinfo.npy", total_info)
            end = time.time()
            duration = end - begin
            # print("cupy save size ",size," bandwidth is", size/duration)
            Inputfile = [path, tensor_shape, tensor_type, size]
        if  bestmethod == "npy":
            # print("save as Numpy")
            Inputnumpy = np.array(Inputtensor)
            path = path + ".npy"
            np.save(path, Inputnumpy)
            Inputfile = [path, tensor_shape, tensor_type, size]
        if  bestmethod == "pt":
            # print("save as Torch")
            path = path + ".pt"
            torch.save(Inputtensor, path)
            Inputfile = [path, tensor_shape, tensor_type, size]
        return Inputfile

    #判断是否是读模式，如果是进入读模式，否则为写模式
    if type == "r":
        if size in readdic:
            if len(readdic[size]) == totalmethods:
                readdic_bestmethod = readdic[size][-1]
                if readdic_bestmethod == "cpy":
                    # print("save as cupy")
                    begin = time.time()
                    tensor_info = []
                    total_info = []
                    #cupy暂时不能处理bool和带有梯度的GPU张量
                    if tensor_type is not torch.bool:
                        outputlist = list(Inputtensor.values())
                        outfile = outputlist[0].view(-1)  #展开 
                        tensor_info.append([outputlist[0].shape, outputlist[0].dtype, len(outputlist[0].view(-1))])
                        for i in outputlist[1:]:
                            outfile = torch.cat((outfile,i.view(-1))) #拼接成一维数组
                            tensor_info.append([i.shape, i.dtype, len(i.view(-1))])
                        Inputcupy = cupy.asarray(outfile, order='C')
                        total_info = [outfile.shape,outfile.dtype]
                    else:
                        Inputcupy = cupy.asarray(Inputtensor.cpu())
                    path = path + ".cpy"
                    f = kvikio.CuFile(path, "w")
                    f.write(Inputcupy)
                    f.close()
                    #保存张量字典的名字和对应的张量类型信息
                    np.save(path + "keys.npy", list(Inputtensor.keys()))
                    np.save(path + "tensorinfo.npy", tensor_info)
                    np.save(path + "totalinfo.npy", total_info)
                    end = time.time()
                    duration = end - begin
                    # print("cupy save size ",size," bandwidth is", size/duration)
                    Inputfile = [path, tensor_shape, tensor_type, size]
                if readdic_bestmethod == "npy":
                    # print("save as Numpy")
                    Inputnumpy = Inputtensor.cpu().numpy()
                    path = path + ".npy"
                    np.save(path, Inputnumpy)
                    Inputfile = [path, tensor_shape, tensor_type, size]
                if readdic_bestmethod == "pt":
                    # print("save as Torch")
                    path = path + ".pt"
                    torch.save(Inputtensor, path)
                    Inputfile = [path, tensor_shape, tensor_type, size]
                # if readdic_bestmethod == "dali":
                #     # print("save as DALI")
                #     Inputnumpy = Inputtensor.cpu().numpy()
                #     path = path + ".dali"
                #     np.save(path, Inputnumpy)
                #     Inputfile = [path, tensor_shape, tensor_type, size]
                return Inputfile

    if size in writedic:
        if len(writedic[size]) == 1:
            #print("save as Numpy")
            torch.cuda.synchronize()
            begin = time.time()
            Inputnumpy = np.array(Inputtensor)
            path = path + ".npy"
            np.save(path, Inputnumpy)
            torch.cuda.synchronize()
            end = time.time()
            duration = end -begin
            #print("numpy save size ",size," bandwidth is", size/duration)
            writedic[size].append([duration,"npy"])
            Inputfile = [path, tensor_shape, tensor_type, size]

        elif len(writedic[size]) == 2:
            #print("save as Torch")
            torch.cuda.synchronize()
            begin = time.time()
            path = path + ".pt"
            torch.save(Inputtensor, path)
            torch.cuda.synchronize()
            end = time.time()
            duration = end -begin
            #print("torch save size ",size," bandwidth is", size/duration)
            writedic[size].append([duration,"pt"])
            fastmethod =  min(writedic[size])[1]
            writedic[size].append(fastmethod)
            Inputfile = [path, tensor_shape, tensor_type, size]
        elif len(writedic[size]) == totalmethods:
            write_best_method = writedic[size][-1]
            if write_best_method == "cpy":
                # print("save as cupy")
                begin = time.time()
                tensor_info = []
                total_info = []
                #cupy暂时不能处理bool和带有梯度的GPU张量
                if tensor_type is not torch.bool:
                    outputlist = list(Inputtensor.values())
                    outfile = outputlist[0].view(-1)  #展开 
                    tensor_info.append([outputlist[0].shape, outputlist[0].dtype, len(outputlist[0].view(-1))])
                    for i in outputlist[1:]:
                        outfile = torch.cat((outfile,i.view(-1))) #拼接成一维数组
                        tensor_info.append([i.shape, i.dtype, len(i.view(-1))])
                    Inputcupy = cupy.asarray(outfile, order='C')
                    total_info = [outfile.shape,outfile.dtype]
                else:
                    Inputcupy = cupy.asarray(Inputtensor.cpu())
                path = path + ".cpy"
                f = kvikio.CuFile(path, "w")
                f.write(Inputcupy)
                f.close()
                #保存张量字典的名字和对应的张量类型信息
                np.save(path + "keys.npy", list(Inputtensor.keys()))
                np.save(path + "tensorinfo.npy", tensor_info)
                np.save(path + "totalinfo.npy", total_info)
                end = time.time()
                duration = end - begin
                # print("cupy save size ",size," bandwidth is", size/duration)
                Inputfile = [path, tensor_shape, tensor_type, size]
            if  write_best_method == "npy":
                #print("save as Numpy")
                Inputnumpy = np.array(Inputtensor)
                path = path + ".npy"
                np.save(path, Inputnumpy)
                Inputfile = [path, tensor_shape, tensor_type, size]
            if  write_best_method == "pt":
                #print("save as Torch")
                path = path + ".pt"
                torch.save(Inputtensor, path)
                Inputfile = [path, tensor_shape, tensor_type, size]
    else:
        # print("save as cupy")
        begin = time.time()
        tensor_info = []
        total_info = []
        #cupy暂时不能处理bool和带有梯度的GPU张量
        if tensor_type is not torch.bool:
            outputlist = list(Inputtensor.values())
            outfile = outputlist[0].view(-1)  #展开 
            tensor_info.append([outputlist[0].shape, outputlist[0].dtype, len(outputlist[0].view(-1))])
            for i in outputlist[1:]:
                outfile = torch.cat((outfile,i.view(-1))) #拼接成一维数组
                tensor_info.append([i.shape, i.dtype, len(i.view(-1))])
            Inputcupy = cupy.asarray(outfile, order='C')
            total_info = [outfile.shape,outfile.dtype]
        else:
            Inputcupy = cupy.asarray(Inputtensor.cpu())
        path = path + ".cpy"
        f = kvikio.CuFile(path, "w")
        f.write(Inputcupy)
        f.close()
        #保存张量字典的名字和对应的张量类型信息
        np.save(path + "keys.npy", list(Inputtensor.keys()))
        np.save(path + "tensorinfo.npy", tensor_info)
        np.save(path + "totalinfo.npy", total_info)
        end = time.time()
        duration = end - begin
        # print("cupy save size ",size," bandwidth is", size/duration)
        writedic[size] = [[duration,"cpy"]]
        Inputfile = [path,tensor_shape,tensor_type, size]
    flag += 1
    return Inputfile


def load(path, device=torch.device('cuda:1')):
    #inputfile is a list [path, tensor_shape, tensor_type]
    global writedic, readdic, wrdic, flag
    # path = Inputfile[0]
    # size = Inputfile[3]
    size = 1
    filetype = path[-3:]
    #补全readdic和wrdic
    if size in readdic:
        if len(readdic[size]) == totalmethods - 1:
            fastmethod = min(readdic[size])[1]
            readdic[size].append(fastmethod)
            '''
            if len(writedic[size]) == totalmethods:
                wrdic[size] = copy.deepcopy(readdic[size][:-1])
                for i in range(totalmethods - 1):
                    for j in range(totalmethods - 1):
                        if wrdic[size][i][1] == writedic[size][j][1]:
                            wrdic[size][i][0] += writedic[size][j][0]
                fastmethod = min(wrdic[size])[1]
                wrdic[size].append(fastmethod)
            '''

    if filetype == "cpy":
        keylist = []
        valuelist = []
        tensor_info = []
        total_info = []
        fileoff1 = 0
        fileoff2 = 0
        keylist = np.load(path + "keys.npy", allow_pickle = True)
        tensor_info = np.load(path + "tensorinfo.npy", allow_pickle = True)
        total_info = np.load(path + "totalinfo.npy", allow_pickle = True)

        value_all = cupy.asarray(torch.empty(total_info[0], dtype = total_info[1]))
        begin = time.time()
        f = kvikio.CuFile(path, "r")
        f.read(value_all)
        end = time.time()
        duration = end - begin
        #print("读出来的总的:", value_all)

        for i in range(len(tensor_info)):  
            fileoff1 = fileoff2
            fileoff2 +=  tensor_info[i][2]
            value_c = value_all[fileoff1:fileoff2] #不读fileoff2位置的
            #print("读出来的:", value_c)
            value_t = torch.as_tensor(value_c,dtype = tensor_info[i][1], device = device)
            value_t = torch.reshape(value_t,shape = tensor_info[i][0])
            #print("转换的tensor:", value_t,"shape:",value_t.shape)
            valuelist.append(value_t)

        Outputtensor = dict(zip(keylist,valuelist))
        
        #首先判断readdic里有没有对应的size，如果没有，则创建并加入cpy的信息，如果有，则查看里面有没有cpy，如果没有，则加入cpy的信息
        if size not in readdic:
            readdic[size] = [[duration,"cpy"]]
        else:
            Outflag = True
            for i in readdic[size]:
                if "cpy" == i[1]:
                    Outflag = False
            if Outflag:
                readdic[size].append([duration, "cpy"])
        os.remove(path)
        os.remove(path + "keys.npy")
        os.remove(path + "tensorinfo.npy")
        os.remove(path + "totalinfo.npy")
        return Outputtensor

    elif filetype == "npy":
        # if path[-8:-4] != "dali":
        torch.cuda.synchronize()
        begin = time.time()
        tensor = np.load(path, allow_pickle = True)
        torch.cuda.synchronize()
        end = time.time()
        duration = end - begin
        if size not in readdic:
            readdic[size] = [[duration, "npy"]]
        else:
            Outflag = True
            for i in readdic[size]:
                if "npy" == i[1]:
                    Outflag = False
            if Outflag:
                readdic[size].append([duration, "npy"])
        os.remove(path)
        return tensor
    else:
        torch.cuda.synchronize()
        begin = time.time()
        tensor = torch.load(path, map_location=lambda storage, loc: storage.cuda(0))
        torch.cuda.synchronize()
        end = time.time()
        duration = end - begin
        if size not in readdic:
            readdic[size] = [[duration, "pt"]]
        else:
            Outflag = True
            for i in readdic[size]:
                if "pt" == i[1]:
                    Outflag = False
            if Outflag:
                readdic[size].append([duration, "pt"])
        os.remove(path)
        return tensor