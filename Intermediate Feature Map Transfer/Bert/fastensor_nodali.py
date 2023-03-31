import torch
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
BSZ = 64
# @pipeline_def(batch_size=1, num_threads=4, device_id=0)
# def pipe_gds(filename):
#     data = fn.readers.numpy(device='gpu', file_root='.', files=filename, register_buffers = True, cache_header_information=True)
#     return data

def save(Inputtensor,path='myfile',flush = False, type = "w", use_dic = "None" ,policy = "None", batch_size = 64):
    global writedic,readdic,flag, BSZ, wrdic, mydic, refdic
    BSZ = batch_size
    size = sys.getsizeof(Inputtensor.storage())/1024/1024
    tensor_shape = Inputtensor.shape
    tensor_type = Inputtensor.dtype
    Inputfile = []
    # print(flag, "save size is", size)
    # flag += 1
    #判断是否使用给定的字典：
    if use_dic != "None" and not bool(mydic):
        path = use_dic
        with open(path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                mydic[round(float(row[0]),6)] = ast.literal_eval(row[1])

    if mydic:
        #保留6位小数以保证size对齐
        size = round(size, 6)
        # print("size is ", size)
        # print("mydic is", mydic)
        bestmethod = mydic[size][-1]
        if  bestmethod == "cpy":
            begin = time.time()
            if tensor_type is not torch.bool:
                Inputcupy = cupy.array(Inputtensor.data, order='C')
            else:
                Inputcupy = cupy.array(Inputtensor.cpu())
            path = path + ".cpy"
            f = kvikio.CuFile(path, "w")
            f.write(Inputcupy)
            f.close()
            end = time.time()
            duration = end - begin
            # print("cupy save size ",size," bandwidth is", size/duration)
            Inputfile = [path, tensor_shape, tensor_type, size]
        elif  bestmethod == "npy":
            # print("save as Numpy")
            begin = time.time()
            Inputnumpy = Inputtensor.cpu().numpy()
            path = path + ".npy"
            np.save(path, Inputnumpy)
            end = time.time()
            duration = end - begin
            # print("numpy save size ",size," bandwidth is", size/duration)
            Inputfile = [path, tensor_shape, tensor_type, size]
        elif  bestmethod == "pt":
            # print("save as Torch")
            path = path + ".pt"
            begin = time.time()
            torch.save(Inputtensor, path)
            end = time.time()
            duration = end - begin
            # print("torch save size ",size," bandwidth is", size/duration)
            Inputfile = [path, tensor_shape, tensor_type, size]
        return Inputfile

    if policy != "None":
        if policy == "cpy":
            # print("save as cupy")
            torch.cuda.synchronize()
            begin = time.time()
            #cupy暂时不能处理bool和带有梯度的GPU张量
            if tensor_type is not torch.bool:
                if tensor_type is not torch.int64:
                    Inputcupy = cupy.array(Inputtensor.detach(), order='C')
                else:
                    Inputcupy = cupy.from_dlpack(Inputtensor.detach())
            else:
                Inputcupy = cupy.array(Inputtensor.cpu())
            # Inputcupy = cupy.array(Inputtensor.cpu())
            path = path + ".cpy"
            f = kvikio.CuFile(path, "w")
            f.write(Inputcupy)
            f.close()
            torch.cuda.synchronize()
            end = time.time()
            duration = end - begin
            # print("save tensor shape", tensor_shape, "save tensor type", tensor_type)
            # print("cupy save size ",size," bandwidth is", size/duration)
            Inputfile = [path, tensor_shape, tensor_type, size]
        if policy  == "npy":
            # print("save as Numpy")
            begin = time.time()
            Inputnumpy = Inputtensor.cpu().numpy()
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

    # if flush or flag == 2100:
    #     # print("readdic is",readdic)
    #     # print("writedic is", writedic)
    #     writedic = {}
    #     readdic = {}
    #     wrdic = {}
    #     flag = 0
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
            # print("save as cupy")
            begin = time.time()
            if tensor_type is not torch.bool:
                if tensor_type is not torch.int64:
                    Inputcupy = cupy.array(Inputtensor.detach(), order='C')
                else:
                    Inputcupy = cupy.from_dlpack(Inputtensor.detach())
            else:
                Inputcupy = cupy.array(Inputtensor.cpu())
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
            Inputnumpy = Inputtensor.cpu().numpy()
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
                    # Inputcupy = cupy.asarray(Inputtensor.cpu())
                    if tensor_type is not torch.bool:
                        if tensor_type is not torch.int64:
                            Inputcupy = cupy.array(Inputtensor.detach(), order='C')
                        else:
                            Inputcupy = cupy.from_dlpack(Inputtensor.detach())
                    else:
                        Inputcupy = cupy.array(Inputtensor.cpu())
                    path = path + ".cpy"
                    f = kvikio.CuFile(path, "w")
                    f.write(Inputcupy)
                    f.close()
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
            Inputnumpy = Inputtensor.cpu().numpy()
            path = path + ".npy"
            np.save(path, Inputnumpy)
            torch.cuda.synchronize()
            end = time.time()
            duration = end -begin
            #print("numpy save size ",size," bandwidth is", size/duration)
            writedic[size].append([duration,"npy"])
            Inputfile = [path, tensor_shape, tensor_type, size]
        # elif len(writedic[size]) == 2:
        #     #print("save as Numpy（used for DALI）")
        #     torch.cuda.synchronize()
        #     begin = time.time()
        #     Inputnumpy = Inputtensor.cpu().numpy()
        #     path = path + ".dali.npy"
        #     np.save(path, Inputnumpy)
        #     torch.cuda.synchronize()
        #     end = time.time()
        #     duration = end -begin
        #     #print("numpy save size （used for DALI）",size," bandwidth is", size/duration)
        #     writedic[size].append([duration,"dali"])
        #     Inputfile = [path, tensor_shape, tensor_type, size]
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
                # Inputcupy = cupy.asarray(Inputtensor.cpu())
                if tensor_type is not torch.bool:
                    if tensor_type is not torch.int64:
                        Inputcupy = cupy.array(Inputtensor.detach(), order='C')
                    else:
                        Inputcupy = cupy.from_dlpack(Inputtensor.detach())
                else:
                    Inputcupy = cupy.array(Inputtensor.cpu())
                path = path + ".cpy"
                f = kvikio.CuFile(path, "w")
                f.write(Inputcupy)
                f.close()
                end = time.time()
                duration = end - begin
                # print("cupy save size ",size," bandwidth is", size/duration)
                Inputfile = [path, tensor_shape, tensor_type, size]
            if  write_best_method == "npy":
                #print("save as Numpy")
                Inputnumpy = Inputtensor.cpu().numpy()
                path = path + ".npy"
                np.save(path, Inputnumpy)
                Inputfile = [path, tensor_shape, tensor_type, size]
            if  write_best_method == "pt":
                #print("save as Torch")
                path = path + ".pt"
                torch.save(Inputtensor, path)
                Inputfile = [path, tensor_shape, tensor_type, size]
            # if  write_best_method == "dali":
            #     #print("save as Numpy (for dali)")
            #     #DALI不具备写能力
            #     Inputnumpy = Inputtensor.cpu().numpy()
            #     path = path + ".npy"
            #     np.save(path, Inputnumpy)
            #     Inputfile = [path, tensor_shape, tensor_type, size]
    else:
        #print("save as cupy")
        # print("save as cupy")
        begin = time.time()
        # Inputcupy = cupy.asarray(Inputtensor.cpu())
        if tensor_type is not torch.bool:
            if tensor_type is not torch.int64:
                Inputcupy = cupy.array(Inputtensor.detach(), order='C')
            else:
                Inputcupy = cupy.from_dlpack(Inputtensor.detach())
        else:
            Inputcupy = cupy.array(Inputtensor.cpu())
        path = path + ".cpy"
        f = kvikio.CuFile(path, "w")
        f.write(Inputcupy)
        f.close()
        end = time.time()
        duration = end - begin
        # print("cupy save size ",size," bandwidth is", size/duration)
        writedic[size] = [[duration,"cpy"]]
        Inputfile = [path,tensor_shape,tensor_type, size]
    return Inputfile


def load(Inputfile, device=torch.device('cuda:0')):
    global writedic, readdic, wrdic, flag, mydic
    path = Inputfile[0]
    size = Inputfile[3]
    filetype = path[-3:]
    tensor_shape = Inputfile[1]
    tensor_type = Inputfile[2]
    # print(flag, "load size is", size)
    # flag += 1
    #补全readdic和wrdic
    if size in readdic:
        if len(readdic[size]) == totalmethods - 1:
            fastmethod = min(readdic[size])[1]
            readdic[size].append(fastmethod)
            if len(writedic[size]) == totalmethods:
                wrdic[size] = copy.deepcopy(readdic[size][:-1])
                for i in range(totalmethods - 1):
                    for j in range(totalmethods - 1):
                        if wrdic[size][i][1] == writedic[size][j][1]:
                            wrdic[size][i][0] += writedic[size][j][0]
                fastmethod = min(wrdic[size])[1]
                wrdic[size].append(fastmethod)
                # print(wrdic[size])
            # print("writedic is", writedic[size])
            # print("readdic is", readdic[size])
            # print("wrdic is", wrdic[size])

    if filetype == "cpy":
        # print("load tensor shape is", tensor_shape, "load tensor type is",tensor_type)
        torch.cuda.synchronize()
        begin = time.time()
        temptensor = torch.empty(tensor_shape , dtype=tensor_type)
        unpackfeatures_c = cupy.asarray(temptensor)
        f = kvikio.CuFile(path, "r")
        f.read(unpackfeatures_c)
        f.close()
        tensor = torch.as_tensor(unpackfeatures_c, device=device)
        torch.cuda.synchronize()
        end = time.time()
        duration = end - begin
        # print("kvikio load size ", size, " bandwidth is", size / duration)
        # print("flag is",flag, "load tensor is", tensor)
        # flag += 1
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
        return tensor

    elif filetype == "npy":
        # if path[-8:-4] != "dali":
        torch.cuda.synchronize()
        begin = time.time()
        tensor = torch.from_numpy(np.load(path)).to(device)
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
        # print("numpy load size ", size, " bandwidth is", size / duration)
        return tensor

        # else:
        #     p = pipe_gds(filename=path)
        #     p.build()
        #     tensor = torch.empty(tensor_shape , dtype=tensor_type).to(device)
        #     torch.cuda.synchronize()
        #     begin = time.time()
        #     pipe_out = p.run()
        #     nvidia.dali.plugin.pytorch.feed_ndarray(pipe_out[0][0], tensor)
        #     torch.cuda.synchronize()
        #     end= time.time()
        #     duration = end - begin
        #     if size not in readdic:
        #         readdic[size] = [[duration, "dali"]]
        #     else:
        #         Outflag = True
        #         for i in readdic[size]:
        #             if "dali" == i[1]:
        #                 Outflag = False
        #         if Outflag:
        #             readdic[size].append([duration, "dali"])
        #     # print("DALI load size ", size, " bandwidth is", size / duration)
        #     return tensor

    # elif filetype[-2:] == "pt":
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
        # print("Torch load size ", size, " bandwidth is", size / duration)
        return tensor