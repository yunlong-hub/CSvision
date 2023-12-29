'''

'''
from mit_semseg.dataset.ADE_dataset import ADE_TrainDataset, ADE_TestDataset, ADE_ValDataset
from mit_semseg.dataset.VOC_dataset import VOC_TrainDataset,VOC_ValDataset,VOC_TestDataset
from mit_semseg.dataset.Cityscapes_dataset import Cityscapes_TestDataset, Cityscapes_TrainDataset, Cityscapes_ValDataset



def train_dataset(cfg):
    if cfg.DATASET.dataset == "ADE20K":
        return ADE_TrainDataset(
        cfg.DATASET.root_dataset,             # 设置数据集根目录
        cfg.DATASET.list_train,               # 设置训练数据列表
        cfg.DATASET,                          # 传递数据集配置
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu) # 每个gpu上batch数目

    elif cfg.DATASET.dataset == "VOC":
        return VOC_TrainDataset(
        cfg.DATASET.root_dataset,             # 设置数据集根目录
        cfg.DATASET,                          # 传递数据集配置
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    elif cfg.DATASET.dataset == "CITY":
        return Cityscapes_TrainDataset(
        cfg.DATASET.root_dataset,             # 设置数据集根目录
        cfg.DATASET,                          # 传递数据集配置
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu
        )

def val_dataset(cfg,start_idx, end_idx):
    if cfg.DATASET.dataset == "ADE20K":
        return ADE_ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        start_idx=start_idx, end_idx=end_idx)

    elif cfg.DATASET.dataset == "VOC":
        return VOC_ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET,
        start_idx=start_idx, end_idx=end_idx)

    elif cfg.DATASET.dataset == "CITY":
        return Cityscapes_ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET,
        start_idx=start_idx, end_idx=end_idx)


def test_dataset(cfg):
    if cfg.DATASET.dataset == "ADE20K":
        return ADE_TestDataset(
        cfg.list_test,
        cfg.DATASET)

    elif cfg.DATASET.dataset == "VOC":
        return VOC_TestDataset(
        root_dir=cfg.DATASET.root_dataset,
        odgt = cfg.list_test,
        opt = cfg.DATASET)


    elif cfg.DATASET.dataset == "CITY":
        return Cityscapes_TestDataset(
        root_dir=cfg.DATASET.root_dataset,
        odgt = cfg.list_test,
        opt = cfg.DATASET)
