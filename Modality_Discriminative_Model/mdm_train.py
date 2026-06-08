import os

import torch
import torch.nn as nn
import wandb
from torch import optim
from torch.utils.data import DataLoader

from options.train_options import TrainOptions
from utils.tools import *

if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = TrainOptions().parse()
    wandb.init(project="PETLDM_classifier",
               entity="aging",
               name=opt.name,
               config={
                   "learning_rate": opt.lr,
                   "architecture": opt.cls_type,
                   "epoch": opt.epoch_count,

               })
    model = define_Cls(opt.cls_type,
                       class_num=opt.class_num,
                       init_type=opt.init_type,
                       init_gain=opt.init_gain,
                       pretrained_pth=opt.pretrained_pth,
                       gpu_ids=opt.gpu_ids)
    epochs = opt.epoch_count
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    if opt.down_resolution:
        from utils.ModalitydDataset_down import ModalitydDataset
    else:
        from utils.ModalitydDataset import ModalitydDataset
    train_set = ModalitydDataset(mode="train", tasktype="all")
    val_set = ModalitydDataset(mode="test", tasktype="all")
    print('length train list:', len(train_set))
    print('length val list:', len(val_set))
    train_loader = DataLoader(train_set,
                              batch_size=int(opt.batch_size // 2),
                              num_workers=int(opt.workers // 2),
                              shuffle=True)

    val_loader = DataLoader(val_set,
                            batch_size=int(opt.batch_size // 2),
                            num_workers=int(opt.workers // 2),
                            shuffle=False)
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    train_data(model, train_loader, val_loader,
               epochs, optimizer,
               criterion, expr_dir, opt.print_freq,
               opt.save_epoch_freq,
               torch.device("cuda"))
    wandb.finish()
