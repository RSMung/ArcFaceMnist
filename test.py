import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.utils as vutils

@torch.no_grad()
def testArcFaceLoss():
    """
    train_feats (_type_): 注册集特征   [train_num, feats_dim]
    train_labels (_type_): 注册集标签   [train_num]
    query_feats (_type_): 查询集特征   [query_num, feats_dim]
    query_labels (_type_): 查询集标签   [query_num]
    """
    train_num = 2
    feats_dim = 512
    query_num = 2
    train_feats = torch.randn((train_num, feats_dim))
    train_labels = torch.randn((train_num))
    query_feats = torch.randn((query_num, feats_dim))
    query_labels = torch.randn((query_num))
    # 将数据移动到 GPU（如果可用）
    train_feats, train_labels = train_feats, train_labels
    query_feats, query_labels = query_feats, query_labels
    # train_feats, train_labels = train_feats.cuda(), train_labels.cuda()
    # query_feats, query_labels = query_feats.cuda(), query_labels.cuda()

    # print(f"query_feats shape:{query_feats.shape}")
    # print(f"train_feats shape:{train_feats.shape}")

    query_feats = torch.flatten(query_feats)
    print(f"query_feats shape:{query_feats.shape}")








if __name__ == "__main__":
    # import os
    # gpu_id = 3
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    testArcFaceLoss()