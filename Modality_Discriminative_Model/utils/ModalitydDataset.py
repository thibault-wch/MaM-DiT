import pickle

import monai.transforms as mtransforms
import torch


def get_individual_list(ori_dict):
    final_list=[]
    for item in ori_dict:
        if item['av45'] != None:
            final_list.append(
                {'mri': item['t1'],
                 'pet': item['av45'],
                 'label': 1})
        if item['fdg'] != None:
            final_list.append(
                {'mri': item['t1'],
                 'pet': item['fdg'],
                 'label': 2})
        if item['mk'] != None:
            final_list.append(
                {'mri': item['t1'],
                 'pet': item['mk'],
                 'label': 3})
    return final_list


class ModalitydDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train", tasktype="all"):
        # basic initialize
        self.basic_transform = mtransforms.Compose(
            [mtransforms.LoadImaged(keys=["mri", "pet"], image_only=True),
             mtransforms.EnsureChannelFirstd(keys=["mri", "pet"]),
             mtransforms.SqueezeDimd(keys=["mri", "pet"]),
             mtransforms.EnsureChannelFirstd(keys=["mri", "pet"]),
             mtransforms.EnsureTyped(keys=["mri", "pet"]),
             mtransforms.ScaleIntensityRangePercentilesd(keys=["mri"], lower=0, upper=99, b_min=-1.0,
                                                         b_max=1.0, clip=True, relative=False),
             mtransforms.ScaleIntensityRanged(keys=["pet"], a_min=0, a_max=2.6,
                                              b_min=-1.0, b_max=1.0, clip=True, ),
             mtransforms.SpatialCropd(keys=["mri", "pet"], roi_center=(128, 128, 128), roi_size=(192, 224, 192)),
             ])

        self.transform = mtransforms.OneOf(
            [
                mtransforms.Compose([]),
                # randflip
                mtransforms.RandFlipd(keys=["mri", "pet"], prob=1, spatial_axis=0),
                mtransforms.RandFlipd(keys=["mri", "pet"], prob=1, spatial_axis=1),
                mtransforms.RandFlipd(keys=["mri", "pet"], prob=1, spatial_axis=2),
                mtransforms.RandFlipd(keys=["mri", "pet"], prob=1),
                # randtranslate
                mtransforms.RandAffined(keys=["mri", "pet"], prob=1.0, translate_range=(60, 0, 0),
                                        padding_mode='border'),
                mtransforms.RandAffined(keys=["mri", "pet"], prob=1.0, translate_range=(0, 60, 0),
                                        padding_mode='border'),
                mtransforms.RandAffined(keys=["mri", "pet"], prob=1.0, translate_range=(0, 0, 60),
                                        padding_mode='border'),
                mtransforms.RandAffined(keys=["mri", "pet"], prob=1.0, padding_mode='border'),
            ]
            , weights=(0.5, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625)
        )
        self.train_pairs = pickle.load(open('train.pkl', 'rb'))
        self.val_pairs = pickle.load(open('val.pkl', 'rb'))
        self.test_pairs = pickle.load(open('test.pkl', 'rb'))
        self.mode = mode
        self.tasktype = tasktype
        if self.mode == "train":
            final_list = get_individual_list(self.train_pairs)
        if self.mode == 'val':
            final_list = get_individual_list(self.val_pairs)
        if self.mode == 'test':
            final_list = get_individual_list(self.test_pairs)
        self.imgs = final_list

    def __getitem__(self, index):
        if self.mode == 'train':

            item = self.transform(
                self.basic_transform(
                    self.imgs[index]
                ))

        else:
            item =self.basic_transform(
                    self.imgs[index]
                )

        return item

    def __len__(self):
        #  the length of dataset
        return len(self.imgs)
