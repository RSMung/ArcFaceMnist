
import torch
import torch.nn as nn
import os

from global_utils import get_rootdir_path
from model.resnet import Resnet18, Resnet18_softmax, Resnet34, Resnet50



def define_model(
        dataset_name, model_name, n_class, 
        ckp_time_stamp=None, 
        # img_size=128
    )->nn.Module:
    """
    define the model structure and init the weights
    Args:
        dataset_name (str)
        model_name (str)
        n_class (int): the number of class
        ckp_time_stamp (str): the checkpoints time stamp of model
        test_flag (bool): src_only, adda, and cyclegan+adda
    """
    # 定义模型结构
    # print(f"model_name:{model_name}")
    if model_name == "resnet18":
        cls_model = Resnet18()
    elif model_name == "resnet18_softmax":
        cls_model = Resnet18_softmax(n_class)
    elif model_name == "resnet34":
        cls_model = Resnet34()
    elif model_name == "resnet50":
        cls_model = Resnet50()
    # elif model_name == "vgg16_4":
    #     cls_model = Vgg16_4(n_class)
    else:
        raise RuntimeError(f"model_name:{model_name} is invalid")
    
    # 加载预训练权重 的函数
    def init_model(net:nn.Module, ckp_path=None):
        if ckp_path is not None:
            print(f"The model has been loaded:{ckp_path}")
            net.load_state_dict(torch.load(ckp_path))
        net.cuda()
        return net
    
    # ckp_time_stamp 不为空时加载预训练权重
    if ckp_time_stamp is not None:
        ckp_path = os.path.join(
                get_rootdir_path(), "ckp", ckp_time_stamp,
                dataset_name + "_" + model_name + "_" + ckp_time_stamp
            )
    else:
        ckp_path = None

    cls_model = init_model(
        cls_model,
        ckp_path=ckp_path
    )
    return cls_model