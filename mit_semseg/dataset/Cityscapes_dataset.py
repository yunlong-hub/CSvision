import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

label_mapping = {
    0: 19,  # unlabeled
    1: 19,  # ego vehicle
    2: 19,  # rectification border
    3: 19,  # out of roi
    4: 19,  # static
    5: 19,  # dynamic
    6: 19,  # ground
    7: 0,    # road
    8: 1,    # sidewalk
    9: 19,  # parking
    10: 19, # rail track
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    14: 19, # guard rail
    15: 19, # bridge
    16: 19, # tunnel
    17: 5,   # pole
    18: 19, # polegroup
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 19, # caravan
    30: 19, # trailer
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
    -1: 19  # license plate
}
def convert_label(label, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')
    return im.resize(size, resample)

class Cityscapes_BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, opt, **kwargs):
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        self.padding_constant = opt.padding_constant
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self.root_dir = root_dir
        self.list_sample = []


    def img_transform(self, img):
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        segm = torch.from_numpy(np.array(segm)).long() - 1  #在 cityscapes中-1，使得unlabeled=-1，有效标签在0-32
        return segm

    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def load_images(self, split, max_sample=-1, start_idx=-1, end_idx=-1):
        split_dir = os.path.join(self.root_dir, 'gtCoarse', split)
        for city in sorted(os.listdir(split_dir)):
            city_folder = os.path.join(split_dir, city)
            for file_name in sorted(os.listdir(city_folder)):
                if file_name.endswith('_gtCoarse_labelIds.png'):
                    mask_path = os.path.join(city_folder, file_name)
                    image_file_name = file_name.replace('_gtCoarse_labelIds.png', '_leftImg8bit.png')
                    image_path = os.path.join(self.root_dir, 'leftImg8bit', split, city, image_file_name)
                    img = Image.open(image_path)
                    width, height = img.size
                    self.list_sample.append(
                        {'fpath_img': image_path, 'fpath_segm': mask_path, 'width': width, 'height': height})
        # 按照要求划分数据集
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

class Cityscapes_TrainDataset(Cityscapes_BaseDataset):
    def __init__(self, root_dir, opt, batch_per_gpu=1, **kwargs):
        super(Cityscapes_TrainDataset, self).__init__(root_dir, opt, **kwargs)
        self.batch_per_gpu = batch_per_gpu
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu
        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]
        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False
        self.load_images('train')

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records


    def __getitem__(self, index):
        # similar to __getitem__ in ADE_TrainDataset
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

            # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), \
                self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(this_record['fpath_img'])
            segm_path = os.path.join(this_record['fpath_segm'])

            img = Image.open(image_path).convert('RGB')
            segm = Image.open(segm_path)
            # # 转换标签,像素点的id到实际的trainid，暂时无用，因为此次训练为-1～33类别的训练
            # segm = np.array(segm)
            # segm = convert_label(segm)
            # return img, Image.fromarray(segm)

            assert (segm.mode == "L")
            assert (img.size[0] == segm.size[0])
            assert (img.size[1] == segm.size[1])

            # random_flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

            # further downsample seg label, need to avoid seg label misalignment
            segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(
                segm_rounded,
                (segm_rounded.size[0] // self.segm_downsampling_rate, \
                 segm_rounded.size[1] // self.segm_downsampling_rate), \
                interp='nearest')

            # image transform, to torch float tensor 3xHxW
            img = self.img_transform(img)

            # segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)

            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output


    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass

class Cityscapes_ValDataset(Cityscapes_BaseDataset):
    def __init__(self, root_dir, opt, **kwargs):
        super(Cityscapes_ValDataset, self).__init__(root_dir,  opt, **kwargs)
        self.load_images('val',**kwargs)

    def __getitem__(self, index):
        # similar to __getitem__ in ADE_ValDataset
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join( this_record['fpath_img'])
        segm_path = os.path.join(this_record['fpath_segm'])
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)
        # # 转换标签,像素点的id到实际的trainid
        # segm = np.array(segm)
        # segm = convert_label(segm)
        # return img, Image.fromarray(segm)

        assert (segm.mode == "L")
        assert (img.size[0] == segm.size[0])
        assert (img.size[1] == segm.size[1])

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample

class Cityscapes_TestDataset(Cityscapes_BaseDataset):
    def __init__(self, root_dir, odgt, opt, **kwargs):
        super(Cityscapes_TestDataset, self).__init__(root_dir, opt, **kwargs)
        self.parse_input_list(odgt, **kwargs)

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image
        image_path = this_record['fpath_img']
        img = Image.open(image_path).convert('RGB')
        ori_width, ori_height = img.size
        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)
            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)
            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')
            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)
        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = this_record['fpath_img']
        return output


    def __len__(self):
        return self.num_sample

