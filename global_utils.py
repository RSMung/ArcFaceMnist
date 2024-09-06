import torch
import torch.nn as nn
import numpy as np
import random
import os

# 固定随机数种子
# set the random seed
def prepareEnv(seed = 1):
    import torch.backends.cudnn as cudnn

    # controls whether cuDNN is enabled. cudnn could accelerate the training procedure
    # cudnn.enabled = False
    cudnn.enabled = True
    # 使得每次返回的卷积算法是一样的
    # if True, causes cuDNN to only use deterministic convolution algorithms
    cudnn.deterministic = True
    # 如果网络的输入数据维度或类型上变化不大, 可以增加运行效率
    # 自动寻找最适合当前的高效算法,优化运行效率
    # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest
    # cudnn.benchmark = False
    cudnn.benchmark = True
    

    """
    在需要生成随机数据的实验中，每次实验都需要生成数据。
    设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。
    """
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数
    torch.cuda.manual_seed_all(seed)  # 给所有GPU设置
    np.random.seed(seed)
    random.seed(seed)



class ParamsParent:
    """
    各个参数类的父类
    """
    def __repr__(self):
        # 直接打印这个类时会调用这个函数, 打印返回的输出的字符串
        str_result = f"---{self.__class__.__name__}---\n"
        # 剔除带__的属性
        # dir(self.__class__)会返回属性的有序列表
        # self.__dir__()返回属性列表, 与前者的区别是不会排序
        for attr in self.__dir__():
            if not attr.startswith('__'):
                str_result += "{}: {}\n".format(attr, self.__getattribute__(attr))
        str_result += "------------------\n"
        return str_result
    
def get_rootdir_path():
    # 当前文件的路径
    current_path = os.path.abspath(__file__)

    # 当前文件所在的目录
    root_dir = os.path.dirname(current_path)
    return root_dir
