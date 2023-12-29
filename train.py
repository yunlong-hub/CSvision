# System libs
import os               # 导入操作系统接口模块
import time             # 导入时间模块
# import math          # 导入数学模块（当前未使用，注释掉）
import random           # 导入生成随机数的模块
import argparse         # 导入解析命令行参数的模块
from distutils.version import LooseVersion  # 导入用于版本号比较的模块
from packaging import version
# Numerical libs
import torch            # 导入PyTorch模块
import torch.nn as nn   # 导入PyTorch神经网络模块
# Our libs
from mit_semseg.config import cfg  # 导入配置文件处理模块
from mit_semseg.models import ModelBuilder, SegmentationModule  # 导入模型构建和分割模块
from mit_semseg.utils import AverageMeter, parse_devices, setup_logger  # 导入工具模块
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback  # 导入神经网络工具模块
# from mit_semseg.all_dataset import train_dataset
from mit_semseg.dataset import train_dataset
# 训练一个周期
def train(segmentation_module, iterator, optimizers, history, epoch, cfg):
    # 初始化一些度量和计时器
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    # 设置模型为训练模式
    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # 主循环开始
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # 加载一批数据
        batch_data = next(iterator)
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # 调整学习率
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)

        # 前向传播
        loss, acc = segmentation_module(batch_data)
        loss = loss.mean()
        acc = acc.mean()

        # 反向传播
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # 测量消耗的时间
        batch_time.update(time.time() - tic)
        tic = time.time()

        # 更新平均损失和准确率
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # 如果需要，显示训练信息
        if i % cfg.TRAIN.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average()))  # 格式化输出当前训练批次的信息

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters  # 计算分数形式的epoch
            history['train']['epoch'].append(fractional_epoch)  # 将epoch添加到训练历史
            history['train']['loss'].append(loss.data.item())  # 将损失添加到训练历史
            history['train']['acc'].append(acc.data.item())  # 将准确率添加到训练历史


def checkpoint(nets, history, cfg, epoch):
    print('Saving checkpoints...')  # 打印保存检查点的信息
    (net_encoder, net_decoder, crit) = nets  # 解包网络模型和损失函数

    dict_encoder = net_encoder.state_dict()  # 获取编码器的状态字典
    dict_decoder = net_decoder.state_dict()  # 获取解码器的状态字典

    # 保存训练历史和模型的状态
    torch.save(history, '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(dict_encoder, '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(dict_decoder, '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))

def group_weight(module):
    group_decay = []       # 定义需要权重衰减的参数组
    group_no_decay = []    # 定义不需要权重衰减的参数组
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)   # 线性层的权重参数加入衰减组
            if m.bias is not None:
                group_no_decay.append(m.bias)  # 线性层的偏置参数加入不衰减组
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)   # 卷积层的权重参数加入衰减组
            if m.bias is not None:
                group_no_decay.append(m.bias)  # 卷积层的偏置参数加入不衰减组
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)  # 批归一化层的权重参数加入不衰减组
            if m.bias is not None:
                group_no_decay.append(m.bias)  # 批归一化层的偏置参数加入不衰减组

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)  # 校验参数总数
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]  # 创建参数组
    return groups  # 返回分组的参数

def create_optimizers(nets, cfg):
    (net_encoder, net_decoder, crit) = nets  # 解包网络模型和损失函数
    # 创建编码器和解码器的优化器
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_encoder, optimizer_decoder)  # 返回创建的优化器

def adjust_learning_rate(optimizers, cur_iter, cfg):
    # 动态调整学习率
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder  # 更新编码器的学习率
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder  # 更新解码器的学习率

def main(cfg, gpus):
    # 构建网络模型
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),  # 构建编码器，使用配置文件中指定的架构
        fc_dim=cfg.MODEL.fc_dim,              # 设置全连接层的维度
        weights=cfg.MODEL.weights_encoder)    # 加载预训练权重
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),  # 构建解码器，使用配置文件中指定的架构
        fc_dim=cfg.MODEL.fc_dim,              # 设置全连接层的维度
        num_class=cfg.DATASET.num_class,      # 设置分类的类别数
        weights=cfg.MODEL.weights_decoder)    # 加载预训练权重

    crit = nn.NLLLoss(ignore_index=-1)       # 创建负对数似然损失函数，忽略特定索引

    # 根据配置决定是否使用深度监督
    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg.TRAIN.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit)

    # 设置训练数据集和加载器
    # dataset_train = TrainDataset(
    #     cfg.DATASET.root_dataset,             # 设置数据集根目录
    #     cfg.DATASET.list_train,               # 设置训练数据列表
    #     cfg.DATASET,                          # 传递数据集配置
    #     batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)  # 设置每个GPU的批次大小，子批次的大小

    dataset_train = train_dataset(cfg)


    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),                # 设置批次大小为GPU数量
        shuffle=False,                       # 设置不随机洗牌
        collate_fn=user_scattered_collate,   # 设置数据整合函数
        num_workers=cfg.TRAIN.workers,       # 设置加载数据时的工作线程数
        drop_last=True,                      # 设置丢弃最后的不完整批次
        pin_memory=True)                     # 设置固定内存
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))  # 打印每个epoch的迭代次数

    # 创建数据加载器迭代器
    # iterator_train = iter(loader_train)

    # 如果使用多个GPU，设置数据并行
    if len(gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=gpus)
        # 为同步批归一化设置回调函数
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()  # 将模型加载到GPU上

    # 设置优化器
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)

    # 主训练循环
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        # 在每个 epoch 开始时重新创建迭代器
        iterator_train = iter(loader_train)
        # 对每个epoch进行训练，并保存检查点
        train(segmentation_module, iterator_train, optimizers, history, epoch+1, cfg)
        checkpoint(nets, history, cfg, epoch+1)

    print('Training Done!')  # 训练完成

# 程序入口
if __name__ == '__main__':
    # 确保PyTorch版本符合要求
    assert version.parse(torch.__version__) >= version.parse('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",                             # 指定配置文件参数
        default="config/city-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",                            # 指定使用的GPU参数
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",                              # 允许修改配置的额外选项
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # 合并配置文件和命令行参数
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # 设置日志记录器
    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # 检查并创建输出目录
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputting checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # 如果有指定起始epoch，加载相应的权重
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "Checkpoint does not exist!"

    # 解析并设置GPU
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu
    # 计算训练的最大迭代次数
    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    # 设置初始学习率
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder
    # 设置随机种子以保证结果可复现
    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    # 调用主函数开始训练
    main(cfg, gpus)
