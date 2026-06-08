
import torch
import pickle
import monai.transforms as mtransforms
def get_individual_list(ori_dict):
    av45_list = []
    fdg_list = []
    mk_list = []
    for item in ori_dict:
        if item['av45'] != None:
            av45_list.append({'mri': item['t1'],
                              'pet': item['av45'],
                              'tasktype': 0})
        if item['fdg'] != None:
            fdg_list.append({'mri': item['t1'],
                             'pet': item['fdg'],
                             'tasktype': 1})
        if item['mk'] != None:
            mk_list.append({'mri': item['t1'],
                            'pet': item['mk'],
                            'tasktype': 2})
    return av45_list, fdg_list, mk_list


class PETDataset(torch.utils.data.Dataset):
    def __init__(self,mode="train",tasktype="all"):
        # basic initialize
        self.basic_transform = mtransforms.Compose(
                [mtransforms.LoadImaged(keys=["mri", "pet"], image_only=True),
                 mtransforms.EnsureChannelFirstd(keys=["mri", "pet"]),
                 mtransforms.SqueezeDimd(keys=["mri", "pet"]),
                 mtransforms.EnsureChannelFirstd(keys=["mri", "pet"]),
                 mtransforms.EnsureTyped(keys=["mri", "pet"]),
                 mtransforms.ScaleIntensityRangePercentilesd(keys=["mri"], lower=0, upper=99, b_min=-1.0,
                                                             b_max=1.0, clip=True, relative=False),
                 mtransforms.ScaleIntensityRanged(keys=["pet"], a_min=0, a_max=2.6, b_min=-1.0, b_max=1.0, clip=True, ),
                 mtransforms.SpatialCropd(keys=["mri", "pet"],roi_center=(128,128,128),roi_size=(192, 224, 192)),
                 ])

        self.transform1 = mtransforms.OneOf(
            [
                mtransforms.Resized(keys=["mri", "pet"], spatial_size=(128, 128, 128)),
                mtransforms.RandSpatialCropd(keys=["mri", "pet"], roi_size=(128, 128, 128))
            ]
            , weights=(0.5, 0.5)
        )

        self.transform2=mtransforms.OneOf(
                [
                mtransforms.Compose([]),
                # randflip
                mtransforms.RandFlipd(keys=["mri", "pet"],prob=1,spatial_axis=0),
                mtransforms.RandFlipd(keys=["mri", "pet"],prob=1,spatial_axis=1),
                mtransforms.RandFlipd(keys=["mri", "pet"],prob=1,spatial_axis=2),
                mtransforms.RandFlipd(keys=["mri", "pet"],prob=1),
                # randtranslate
                mtransforms.RandAffined(keys=["mri", "pet"],prob=1.0,translate_range=(60, 0, 0),padding_mode='border'),
                mtransforms.RandAffined(keys=["mri", "pet"],prob=1.0,translate_range=(0, 60, 0),padding_mode='border'),
                mtransforms.RandAffined(keys=["mri", "pet"],prob=1.0,translate_range=(0, 0, 60),padding_mode='border'),
                # mtransforms.RandAffined(keys=["mri", "pet"],prob=1.0,padding_mode='border'),
                ]
                , weights=(0.5625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625)
            )
        self.train_pairs = pickle.load(open('/cpfs01/projects-HDD/cfff-7abceac4e328_HDD/dhm_41310/chwang/Logs/train.pkl', 'rb'))
        self.test_pairs = pickle.load(open('/cpfs01/projects-HDD/cfff-7abceac4e328_HDD/dhm_41310/chwang/Logs/test.pkl', 'rb'))
        self.mode = mode
        self.tasktype = tasktype
        if self.mode == "train":
            av45_list, fdg_list, mk_list = get_individual_list(self.train_pairs)
        if self.mode == 'test':
            av45_list, fdg_list, mk_list = get_individual_list(self.test_pairs)
        if self.tasktype == 'all':
            self.imgs = av45_list + fdg_list + mk_list
        elif self.tasktype == 'av45':
            self.imgs = av45_list
        elif self.tasktype == 'fdg':
            self.imgs = fdg_list
        elif self.tasktype == 'mk':
            self.imgs = mk_list

    def __getitem__(self, index):
        # extract specific data <important part>
        if self.mode == 'train':
            item = self.transform2(self.basic_transform(
                    self.imgs[index]

            ))
        else:
            item=self.basic_transform(
                    self.imgs[index]

            )
        return item

    def __len__(self):
        #  the length of dataset
        return len(self.imgs)

