import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from EasyLossUtil.global_utils import ParamsParent, checkDir, formatSeconds
from EasyLossUtil.easyLossUtil import EasyLossUtil

import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
from tqdm import tqdm
import sys
import signal
from datetime import timedelta

from data.getDataloader import get_dataloader
from global_utils import prepareEnv
from model.get_model import define_model


def build_src_ckp_path(ckp_time_stamp, dataset_name, model_name):
    assert ckp_time_stamp is not None
    assert dataset_name is not None
    assert model_name is not None
    # 当前文件的路径
    current_path = os.path.abspath(__file__)
    # 当前文件所在的目录
    root_dir = os.path.dirname(current_path)
    # the path for saving model checkpoints
    ckp_root_path = os.path.join(root_dir, "ckp", ckp_time_stamp)
    checkDir(ckp_root_path)
    cls_model_model_ckp_path = os.path.join(
        ckp_root_path,
        dataset_name + "_" + model_name + "_" + ckp_time_stamp
    )
    # the path for saving loss data
    loss_root_path = os.path.join(root_dir, "loss", "loss_" + ckp_time_stamp)
    return cls_model_model_ckp_path, loss_root_path


class TrainResnetParams(ParamsParent):
    # gpu_id = 0
    # gpu_id = 1
    gpu_id = 2
    # gpu_id = 3

    dataset_name = "mnist"
    n_class = 10
    img_size = 128
    proportion = None   # mnist 数据集不需要比例参数, 默认 50000:10000:10000

    # batch_size = 48
    batch_size = 128
    lr = 5e-5

    total_epochs = 1000
    # total_epochs = 1
    early_stop_epochs = 20

    backbone_type = "resnet18"
    loss_fuc_type = "softmax"
    model_name = backbone_type + "_" + loss_fuc_type

    use_tqdm = False
    # use_tqdm = True

    # 是否快速调试
    quick_debug = False
    # quick_debug = True

    ckp_time_stamp = "2024-09-04_09-01"   # 实验 2
    
    model_ckp_path, loss_root_path = build_src_ckp_path(
        ckp_time_stamp,
        dataset_name,
        model_name
    )

    # nohup python -u main.py > ./log/2024-09-04_09-01.txt 2>&1 &
    # 实验 2      1729914
    


class ArcFaceLoss(nn.Module):
    def __init__(self, feat_dim:int, n_class:int, scale=10, margin=0.5):
        """
        Args:
            feat_dim (int): 特征维度
            n_class (int): 类别数量
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.n_class = n_class
        self.scale = scale   # 放缩系数
        self.margin = torch.tensor(margin)   # margin值
        # 最后一个全连接层的权重参数
        self.weight = nn.Parameter(torch.rand(feat_dim, n_class), requires_grad=True)
        nn.init.xavier_uniform_(self.weight) # Xavier 初始化 FC 权重
    
    # def forward(self, feats:torch.Tensor, labels:torch.Tensor):
    #     # 只对输入特征 x_i 的真实类别 y_i 增加margin
    #     # idx_theta_m = (labels == 1)   # [N, n]
    #     # 不变化的位置
    #     idx_theta = (labels == 0).long()   # [N, n]

    #     # 归一化特征以及权重
    #     feats = F.normalize(feats, dim=1)   # [N, feats_dim]
    #     w = F.normalize(self.weight, dim=0)   # [N, n]  n是类别数量
    #     # 二者点乘得到cos_theta
    #     cos_theat = torch.matmul(feats, w)   # [N, n]

    #     # 求得sin_theta
    #     sin_theat = torch.sqrt(1.0 - torch.pow(cos_theat, 2))   # [N, n]
    #     # 计算 cos(theta + m) = cos_theta * cos_m - sin_theta * sin_m
    #     cos_theat_m = cos_theat * torch.cos(self.m) - sin_theat * torch.sin(self.m)   # [N, n]

    #     logits = idx_theta * cos_theat + labels * cos_theat_m
    #     logits = self.s * logits
    #     return logits
    
    def forward(self, feats:torch.Tensor, labels:torch.Tensor):
        # print(feats.shape)
        # print(self.weight.shape)
        # 归一化特征向量以及权重参数，然后计算它们的矩阵乘法
        cos_theta = torch.matmul(F.normalize(feats), F.normalize(self.weight))
        # 防止数值问题
        cos_theta = cos_theta.clip(-1+1e-7, 1-1e-7)
        
        # 计算角度值
        arc_cos = torch.acos(cos_theta)
        # 在特定位置给角度值设定margin值
        M = F.one_hot(labels, num_classes = self.n_class) * self.margin
        # 加上margin矩阵
        arc_cos = arc_cos + M
        
        # 恢复为logits
        cos_theta_2 = torch.cos(arc_cos)
        # 放缩
        logits = cos_theta_2 * self.scale
        return logits
    


def train_procedure(
        params:TrainResnetParams,
        cls_model:nn.Module, 
        train_dataloader,
        val_dataloader,
        # test_dataloader
    ):
    """
    the train procedure for training normal classifier model
    Args:
        params (TrainNormalClsParams): all the parameters
        cls_model (nn.Module): the model we want to train
        arcface_loss_func (nn.Module): arcface module
        train_dataloader: training data
        val_dataloader: validating data
    """
    # ------------------------------
    # -- init tool
    # ------------------------------
    loss_name = ['avg_train_batch_loss', 'val_loss', 'val_acc', 'val_eer']
    # 数据可视化处理工具
    lossUtil = EasyLossUtil(
        loss_name_list=loss_name,
        loss_root_dir=params.loss_root_path
    )

    # -----------------------------------
    # --  setup optimizer and lr_scheduler
    # -----------------------------------
    optimizer = optim.AdamW(
    # optimizer = optim.RMSprop(
        # vgg16_2 only update the params of fc module
        cls_model.parameters(),
        lr=params.lr,
        # weight_decay=0.05
        weight_decay=5e-4
    )
    # Decay LR by a factor of 0.1 every 7 epochs
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

    # ------------------------------
    # --  setup loss function
    # ------------------------------
    ce_loss_func = nn.CrossEntropyLoss()

    #region kill process
    # ----------------------------------------------------------------------
    # 定义一个信号处理函数, 用于处理kill命令的SIGTERM信号, 在退出前保存一次模型
    # ----------------------------------------------------------------------
    def handle_sigterm(signum, frame):
        """
        当使用 kill <进程id> 命令终止程序时会调用这个程序
        当使用 ctrl+c 命令终止程序时会调用这个程序
        Args:
            signum: 一个整数，代表接收到的信号的编号. signal.SIGTERM
            frame: 一个包含有关信号的堆栈信息的对象
        """
        if signum == signal.SIGTERM:
            print("Received kill signal. Performing saving procedure for ckps...")
        elif signum == signal.SIGINT:
            print("Received Ctrl+C signal. Performing saving procedure for ckps...")
        torch.save(
            cls_model.state_dict(), 
            "kill_"+params.model_ckp_path
        )
        print(f"Saving procedure completed:{params.model_ckp_path}")
        sys.exit(0)

    # 注册SIGTERM信号处理函数
    # 将handle_sigterm函数与特定的信号（signal.SIGTERM）相关联
    signal.signal(signal.SIGTERM, handle_sigterm)
    # SIGINT是Ctrl+C传送的中断进程信号
    signal.signal(signal.SIGINT, handle_sigterm)
    #endregion

    # ------------------------------
    # --  train the model
    # ------------------------------
    min_val_loss = None
    no_change_epochs = 0
    avg_train_batch_loss = 0
    batch_num = 0
    for epoch in range(params.total_epochs):
        start_time = time.time()
        # setup network
        cls_model.train()
        # setup the progress bar
        if params.use_tqdm:
            iter_object = tqdm(train_dataloader, ncols=100)
        else:
            iter_object = train_dataloader
        for step, (images, labels) in enumerate(iter_object):
            if params.quick_debug:
                if step > 3:
                    break
            # print(images.shape)
            # print(labels.shape)
            # send images to gpu
            images = images.cuda()
            labels = labels.cuda()

            # # 将labels变成onehot形式
            # one_hot_labels = F.one_hot(labels, params.n_class).cuda()
            # # one_hot_labels = torch.zeros((images.shape[0], params.n_class), device='cuda')
            # # one_hot_labels.scatter_(dim=1, index=labels.view(-1, 1).long(), value=1)

            # zero gradients for optimizer
            optimizer.zero_grad()

            logits = cls_model(images)

            # 计算损失值
            batch_loss = ce_loss_func(logits, labels)

            # 记录损失值
            avg_train_batch_loss += batch_loss.item()
            batch_num += 1

            # 更新参数
            batch_loss.backward()
            optimizer.step()
        # end for iter_object

        # one epoch end

        avg_train_batch_loss /= batch_num

        # print("-----validate model on training set-----")
        # # validate model on training set
        # train_loss, train_acc, train_eer = validate_procedure(
        #         cls_model, arcface_loss_func,
        #         train_dataloader, train_dataloader, 
        #         params
        # )
        # print("-----validate model on validation set-----")
        # validate model on validation set
        val_loss, val_acc, val_eer = validate_procedure(
                cls_model,
                train_dataloader, val_dataloader, 
                params
        )
        # print("-----validate model on test set-----")
        # # validate model on test set
        # test_loss, test_acc, test_eer = validate_procedure(
        #         cls_model, arcface_loss_func,
        #         train_dataloader, test_dataloader, 
        #         params
        # )

        # adjust the learning rate
        step_lr_scheduler.step()

        # display train log
        print(
            f'[{epoch}/{params.total_epochs}]\n'
            f'avg_train_batch_loss: {avg_train_batch_loss:.4f}\n '
            # f'train_acc: {train_acc*100:.2f} %, '
            # f'train_eer: {train_eer*100:.2f} %\n'
            f'val_loss: {val_loss:.4f} , '
            f'val_acc: {val_acc*100:.2f} %, '
            f'val_eer: {val_eer*100:.2f} %\n'
            # f'test_loss: {test_loss:.4f} , '
            # f'test_acc: {test_acc*100:.2f} %, '
            # f'test_eer: {test_eer*100:.2f} %'
        )

        # print(f"avg_train_batch_loss:{avg_train_batch_loss.device}")
        # print(f"val_loss:{val_loss.device}")
        # print(f"val_acc:{val_acc.device}")
        # print(f"val_eer:{val_eer.device}")
        # save the log data
        lossUtil.append(
            loss_name=loss_name,
            loss_data=[
                avg_train_batch_loss, 
                # train_acc, train_eer,
                val_loss, val_acc, val_eer,
                # test_loss, test_acc, test_eer
            ]
        )
        lossUtil.autoSaveFileAndImage()

        # save model parameters
        no_change_epochs += 1
        if min_val_loss is None or val_loss < min_val_loss:
            torch.save(
                cls_model.state_dict(), 
                params.model_ckp_path
            )
            min_val_loss = val_loss
            no_change_epochs = 0
            print('已经保存当前模型')
        
        # the time for this epoch
        end_time = time.time()
        print(f"epoch time cost:   {str(timedelta(seconds=int(end_time-start_time)))}")
        print()

        # early stop
        if no_change_epochs > params.early_stop_epochs:
            print('early stop train model')
            break

    # end all epoch
    return cls_model



@torch.no_grad()
def validate_procedure(
    cls_model:nn.Module, 
    train_dataloader:DataLoader,
    query_dataloader:DataLoader, 
    params:TrainResnetParams,
    save_csv=False
):
    """
    验证环节, 计算模型在指定数据集上的性能
    Args:
        cls_model (nn.Module): 特征提取器
        train_dataloader (DataLoader): 训练集, 即注册集
        query_dataloader (DataLoader): 指定的数据集
        params (TrainResnetParams): 外部参数
    """
    # print("obtain the features and labels of training data")
    train_feats, train_labels = get_feats_labels(cls_model, train_dataloader, params)
    # print("obtain the features and labels of query data")
    query_feats, query_labels = get_feats_labels(cls_model, query_dataloader, params)

    # print("get the identification accuracy on train data")
    acc = identification_procedure(
        train_feats, train_labels, 
        query_feats, query_labels
    )
    # print("get the eer on train data")
    eer = verification_procedure(
        train_feats, train_labels, 
        query_feats, query_labels,
        save_csv=save_csv
    )
    # print("get the value of cross entropy loss")
    ce_loss = get_cross_entropy_loss(
        cls_model, 
        query_dataloader, params
    )
    return ce_loss, acc, eer


@torch.no_grad()
def get_feats_labels(cls_model:nn.Module, dataloader:DataLoader, params:TrainResnetParams):
    # setup the model on eval mode
    cls_model.eval()

    # setup the dataloader
    if params.use_tqdm:
        iter_object = tqdm(dataloader, ncols=100)
    else:
        iter_object = dataloader
    
    all_feats = []
    all_labels = []

    # loop
    for (images, labels) in iter_object:
        images = images.cuda()
        labels = labels.cuda()

        feats = cls_model(images)
        all_feats.append(feats)
        all_labels.append(labels)
    # end for dataloader

    # list to tensor
    all_feats = torch.concat([item for item in all_feats])
    all_labels = torch.concat([item for item in all_labels])
    return all_feats, all_labels


@torch.no_grad()
def get_cross_entropy_loss(cls_model:nn.Module, dataloader:DataLoader, params:TrainResnetParams):
    # setup the model on eval mode
    cls_model.eval()

    # setup the dataloader
    if params.use_tqdm:
        iter_object = tqdm(dataloader, ncols=100)
    else:
        iter_object = dataloader

    ce_entropy_loss = nn.CrossEntropyLoss()

    ce_loss = 0
    batch_num = 0

    # loop
    for (images, labels) in iter_object:
        images = images.cuda()
        labels = labels.cuda()

        logits = cls_model(images)
        batch_loss = ce_entropy_loss(logits, labels)

        ce_loss += batch_loss.item()
        batch_num += 1
    # end for dataloader

    ce_loss /= batch_num
    return ce_loss



@torch.no_grad()
def identification_procedure(
    train_feats, train_labels, 
    query_feats, query_labels
):
    """
    credits: https://github.com/weixu000/DSH-pytorch/blob/906399c3b92cf8222bca838c2b2e0e784e0408fa/utils.py#L58
    查询特征与注册集中的特征匹配, 看和哪个最接近以确定预测标签, 然后基于此计算精度acc
    Args:
        train_feats (_type_): 注册集特征   [train_num, feats_dim]
        train_labels (_type_): 注册集标签   [train_num]
        query_feats (_type_): 查询集特征   [query_num, feats_dim]
        query_labels (_type_): 查询集标签   [query_num]

    Returns:
        float: acc
    """
    correct_num = 0
    query_samples_num = query_feats.size(0)

    # 将数据移动到 GPU（如果可用）
    train_feats, train_labels = train_feats.cuda(), train_labels.cuda()
    query_feats, query_labels = query_feats.cuda(), query_labels.cuda()

    # print(f"query_feats shape:{query_feats.shape}")
    # print(f"train_feats shape:{train_feats.shape}")

    # 计算余弦相似度
    query_feats = F.normalize(query_feats)   # 归一化, 将某一个维度除以那个维度对应的范数(默认是2范数, dim=1)
    train_feats = F.normalize(train_feats)
    cosine_similarity = torch.matmul(query_feats, train_feats.transpose(0,1))   # [query_num, train_num]
    # print(f"cosine_similarity: {cosine_similarity.shape}")

    _, predicted_idx = torch.max(cosine_similarity, dim=1)   # [query_num]

    predicted_labels = train_labels[predicted_idx]

    # 将预测标签与query的真实标签比较
    correct_mask = (query_labels == predicted_labels)
    # print(f"correct_mask: {correct_mask.shape}")

    # 如果检索成功，则计数器加一
    correct_num = correct_mask.sum().item()

    # 计算精度
    # print(f"correct_num: {correct_num}")
    # print(f"num_samples: {num_samples}")
    acc = correct_num / query_samples_num
    # print(f"acc: {acc}")

    return acc


@torch.no_grad()
def verification_procedure(
    train_feats, train_labels, 
    query_feats, query_labels,
    save_csv=False
):
    # query_samples_num = query_feats.size(0)

    # 将数据移动到 GPU（如果可用）
    train_feats, train_labels = train_feats.cuda(), train_labels.cuda()
    query_feats, query_labels = query_feats.cuda(), query_labels.cuda()

    # print(f"train_labels shape: {train_labels.shape}")
    # print(f"query_labels shape: {query_labels.shape}")
    
    # 计算余弦相似度
    query_feats = F.normalize(query_feats)   # 归一化, 将某一个维度除以那个维度对应的范数(默认是2范数, dim=1)
    train_feats = F.normalize(train_feats)
    cosine_similarity = (query_feats @ train_feats.transpose(0,1)).to(torch.float16)   # [query_num, train_num]
    # cosine_distance = 1 - cosine_similarity
    del query_feats, train_feats

    # 构造pairs的真假标签   [query_num, train_num]
    genuine_or_imposter_labels = torch.eq(query_labels[:, None], train_labels[None, :]).to(torch.int8)
    # print(f"genuine_or_imposter_labels: {genuine_or_imposter_labels.shape}")
    del query_labels, train_labels

    # genuine_idxs = torch.where(genuine_or_imposter_labels == 1)
    # genuine_cosine_distance = cosine_similarity[genuine_idxs]

    # imposter_idxs = torch.where(genuine_or_imposter_labels == 0)
    # imposter_cosine_distance = cosine_similarity[imposter_idxs]

    cosine_similarity = cosine_similarity.cpu()
    genuine_or_imposter_labels = genuine_or_imposter_labels.cpu()

    #region my eer method
    # # --------------------------------------------------
    # # - 逐batch处理
    # # --------------------------------------------------
    # batch_size = 1000  # 根据资源消耗情况调整

    # genuine_cosine_similarity = []
    # imposter_cosine_similarity = []

    # # 按行分批处理
    # for i in range(0, cosine_similarity.size(0), batch_size):
    #     # 取出当前批次的 cosine_similarity 和 labels
    #     batch_cosine_similarity = cosine_similarity[i:i + batch_size]
    #     batch_labels = genuine_or_imposter_labels[i:i + batch_size]
        
    #     # 提取 genuine 数据
    #     genuine_mask = (batch_labels == 1)
    #     genuine_cosine_similarity.append(batch_cosine_similarity[genuine_mask])
        
    #     # 提取 imposter 数据
    #     imposter_mask = (batch_labels == 0)
    #     imposter_cosine_similarity.append(batch_cosine_similarity[imposter_mask])

    # # 最后将所有批次的结果拼接起来
    # genuine_cosine_similarity = torch.cat(genuine_cosine_similarity)
    # imposter_cosine_similarity = torch.cat(imposter_cosine_similarity)

    # # 显示矩阵大小
    # print("Genuine Cosine similarity shape:", genuine_cosine_similarity.shape)
    # print("Imposter Cosine similarity shape:", imposter_cosine_similarity.shape)
    # 划分 真-假 piar的距离结束

    # t_low, t_high = torch.min(cosine_similarity), torch.max(cosine_similarity)
    # print(f"t_low:{t_low}")
    # print(f"t_high:{t_high}")

    # del cosine_similarity, genuine_or_imposter_labels

    # # step = 1
    # step = 0.01
    # # thresholds = torch.linspace(start=t_low-step, end=t_high+step, steps=int((t_high - t_low) / step))
    # thresholds = torch.arange(start=t_low-step, end=t_high+2*step, step=step)

    # t_num = thresholds.shape[0]

    # idx_min, t_min, frr_min, far_min, d_min = None, None, None, None, None

    # if save_csv:
    #     # DET 曲线
    #     # 横轴是FAR, 即错误接收负样本概率   FPR
    #     all_far_data = []
    #     # 纵轴是FRR, 即错误拒绝正样本概率
    #     all_frr_data = []

    # for j in range(t_num):
    #     t_j = thresholds[j]
    #     # 真pair的相似度 < 阈值, 被拒绝, 错误地拒绝
    #     fr = torch.sum(genuine_cosine_similarity < t_j)
    #     # 假pair的相似度 > 阈值, 被接受, 错误地接受
    #     fa = torch.sum(imposter_cosine_similarity > t_j)
    #     # 错误拒绝的数量占总的真pair的比例
    #     frr = fr / genuine_cosine_similarity.shape[0]
    #     # 错误接受的数量占总的假pair的比例
    #     far = fa / imposter_cosine_similarity.shape[0]

    #     if save_csv:
    #         # 存储数据
    #         all_far_data.append(far.item())
    #         all_frr_data.append(frr.item())
        
    #     print(f"t_j:{t_j}")
    #     print(f"far:{far}")
    #     print(f"frr:{frr}")
    #     print()

    #     # 寻找最接近的frr和far
    #     d_current = torch.abs(frr - far)
    #     if (idx_min is None) or (d_current < d_min):
    #         idx_min, t_min, frr_min, far_min, d_min = j, t_j, frr, far, d_current

    # eer = ((frr_min + far_min) / 2).item()

    # print(f"frr_min: {frr_min}")
    # print(f"far_min: {far_min}")
    #endregion
    

    #region sklearn roc_curve
    # flatten and convert to numpy matrix
    cosine_similarity = torch.flatten(cosine_similarity).numpy()
    genuine_or_imposter_labels = torch.flatten(genuine_or_imposter_labels).numpy()
    # utilizing the sklearn function
    from sklearn.metrics import roc_curve
    all_far_data, tar, thresholds = roc_curve(genuine_or_imposter_labels, cosine_similarity)
    all_frr_data = 1 - tar
    import numpy as np
    eer = all_far_data[np.nanargmin(np.absolute((all_far_data - all_frr_data)))]
    #endregion


    if save_csv:
        # 保存 det 数据到csv文件
        import os
        import pandas as pd
        # tar_data, far_data = get_roc_data_cdtrans(y_true, y_score)
        current_path = os.path.abspath(__file__)   # 本文件的目录
        root_dir = os.path.dirname(current_path)   # 本文件的父目录
        # tar
        pdOperator_far = pd.DataFrame(
            data={"FAR":all_far_data, "FRR":all_frr_data}
        )
        pdOperator_far.to_csv(
            os.path.join(
                root_dir,
                TrainResnetParams.model_name+"_far_frr_"+TrainResnetParams.dataset_name+ ".csv"
            ),
            index=False, 
            # header=False
        )

    return eer


def trainResnetSoftmaxMain():
    # ------------------------------
    # -- init the env
    # ------------------------------
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(TrainResnetParams.gpu_id)
    prepareEnv()
    # ------------------------------
    # -- init src train params
    # ------------------------------
    params = TrainResnetParams()
    print(params)
    # ------------------------------
    # -- load data
    # ------------------------------
    print("=== load data ===")
    train_dataloader = get_dataloader(
        params.dataset_name, 
        phase="train", 
        img_size=params.img_size, 
        batch_size=params.batch_size,
        proportion=params.proportion
    )
    print(len(train_dataloader.dataset))
    val_dataloader = get_dataloader(
        params.dataset_name, 
        "val", 
        img_size=params.img_size, 
        batch_size=params.batch_size,
        proportion=params.proportion
    )
    print(len(val_dataloader.dataset))
    test_dataloader = get_dataloader(
        params.dataset_name, 
        "test", 
        img_size=params.img_size, 
        batch_size=params.batch_size,
        proportion=params.proportion
    )
    print(len(test_dataloader.dataset))

    # ------------------------------------------
    # -- define the structure of src model
    # ------------------------------------------
    print("=== init model ===")
    # params.dataset_name, params.model_name, params.n_class
    cls_model = define_model(params.dataset_name, params.model_name, params.n_class)
    print(type(cls_model))

    # # ------------------------------
    # # -- train model
    # # ------------------------------
    # print("=== train model ===")
    # cls_model = train_procedure(
    #     params,
    #     cls_model,
    #     train_dataloader, val_dataloader, 
    #     # test_dataloader
    # )

    # ------------------------------
    # -- test best model
    # ------------------------------
    print("test best model")
    # load archive
    archive = torch.load(params.model_ckp_path)
    # 加载进模型
    cls_model.load_state_dict(archive)
    # setup mode
    cls_model.eval()
    # validate model on test set
    test_loss, test_acc, test_eer = validate_procedure(
        cls_model,
        train_dataloader, test_dataloader, 
        params,
        save_csv=True
    )
    print(
        f'test_loss:{test_loss:.4f}, '
        f'test_acc: {test_acc*100:.4f} %, '
        f'test_eer: {test_eer*100:.2f} %'
    )



# def testArcFaceLoss():
#     loss_func = ArcFaceLoss(feat_dim=512, n_class=10)
#     feats = torch.randn((2, 512))
#     a = loss_func(feats)


# if __name__ == "__main__":
#     testArcFaceLoss()