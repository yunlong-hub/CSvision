import matplotlib.pyplot as plt
import pandas as pd
import torch


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description="绘制图像"
    # )
    # parser.add_argument(
    #     "--path",
    #     required=True,
    #     type=str,
    #     help="an history path"
    # )
    # args = parser.parse_args()
    # history = torch.load(args.path)
    # 加载历史数据
    history = torch.load(
        '/data/yunlong/Project/CSvision/ckpt/CITY/city-resnet50dilated-ppm_deepsup/val_history.pth'
        )

    # 下采样数据点
    # 比如每隔10个点取一个点进行绘图
    downsample_factor = 1  # 根据需要调整下采样的密度
    epoch_downsampled = history['val']['epoch'][::downsample_factor]
    IoU_downsampled = history['val']['Mean IoU'][::downsample_factor]
    acc_downsampled = history['val']['acc'][::downsample_factor]

    # 将下采样后的数据转换为Pandas DataFrame
    IoU_data_subsampled = pd.DataFrame({
        'epoch': epoch_downsampled,
        'Iou': IoU_downsampled
    })

    acc_data_subsampled = pd.DataFrame({
        'epoch': epoch_downsampled,
        'acc': acc_downsampled
    })

    # 使用移动平均进行平滑，window是平滑窗口的大小
    window_size = 1  # 根据需要调整窗口大小
    IoU_data_subsampled['IoU_smooth'] = IoU_data_subsampled['Iou'].rolling(window=window_size, min_periods=1).mean()
    acc_data_subsampled['acc_smooth'] = acc_data_subsampled['acc'].rolling(window=window_size, min_periods=1).mean()

    # # 绘制下采样并平滑后的损失曲线
    # plt.figure(figsize=(10, 5))
    # plt.plot(IoU_data_subsampled['epoch'], IoU_data_subsampled['IoU_smooth'], label='IoU')
    # plt.title('Eval_IoU')
    # plt.xlabel('Epoch')
    # plt.ylabel('IoU')
    # plt.legend()
    # plt.show()
    #
    # # 绘制下采样并平滑后的准确率曲线
    # plt.figure(figsize=(10, 5))
    # plt.plot(acc_data_subsampled['epoch'], acc_data_subsampled['acc_smooth'], label='acc')
    # plt.title('Eval_acc')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()


    # 创建一个图形，并包含两个子图（1行2列）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制损失曲线在第一个子图中
    ax1.plot(IoU_data_subsampled['epoch'], IoU_data_subsampled['IoU_smooth'], label='IoU')
    ax1.set_title('Eval IoU', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('IoU', fontsize=14)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)  # 例如，这里将刻度字体大小设置为12
    # 绘制准确率曲线在第二个子图中
    ax2.plot(acc_data_subsampled['epoch'], acc_data_subsampled['acc_smooth'], label='Accuracy')
    ax2.set_title('Eval Accuracy', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.legend(fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=16)  # 也将这个刻度字体大小设置为12

    # 调整子图的布局
    plt.tight_layout()

    # 显示图形
    plt.show()


