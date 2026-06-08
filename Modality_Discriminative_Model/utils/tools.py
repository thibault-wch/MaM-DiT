import time

import numpy as np
import torch
import wandb
from models.Resnet18 import resnet18
from sklearn.metrics import f1_score
from torch.nn import init
from tqdm import tqdm


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, pretrained_pth=None, gpu_ids=''):
    if pretrained_pth != None:
        print('initiaze the model from: {}'.format(pretrained_pth))
        net.load_state_dict(torch.load(pretrained_pth, map_location='cpu'), strict=False)
    else:
        init_weights(net, init_type, gain=init_gain)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net)
        net.cuda()

    return net


def define_Cls(netCls, class_num=4, init_type='normal', init_gain=0.02, pretrained_pth=None, gpu_ids=[]):
    if netCls == 'resnet3d':
        net = resnet18(num_classes=class_num)
    return init_net(net, init_type, init_gain, pretrained_pth, gpu_ids)


def train_data(model, train_dataloaders, valid_dataloaders,
               epochs, optimizer, criterion,
               expr_dir, print_freq, save_epoch_freq,
               device='cpu'):
    start = time.time()
    steps = 0

    for e in tqdm(range(1, epochs + 1)):
        model.train()
        train_loss = 0.
        train_correct_sum = 0.
        train_simple_cnt = 0.
        y_train_true = []
        y_train_pred = []
        train_prob_all = []
        train_label_all = []
        for ii, item in enumerate(tqdm(train_dataloaders)):
            steps += 1
            images = torch.cat((item['mri'], item['pet']), dim=0)
            labels = torch.cat((torch.zeros(int(images.shape[0] // 2)).to(torch.long), item['label']), dim=0)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss_cls = criterion(outputs, labels)
            loss = 1 * loss_cls
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct_sum += (labels.data == train_predicted).sum().item()
            train_simple_cnt += labels.size(0)
            y_train_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
            y_train_pred.extend(np.ravel(np.squeeze(train_predicted.cpu().detach().numpy())).tolist())
            train_prob_all.extend(outputs[:, 1].cpu().detach().numpy())
            train_label_all.extend(labels.cpu())
            wandb.log({"train_loss": loss.item()})

        val_correct_sum = 0
        val_simple_cnt = 0
        val_loss = 0
        y_val_true = []
        y_val_pred = []
        val_prob_all = []
        val_label_all = []
        with torch.no_grad():
            model.eval()
            for ii, item in enumerate(tqdm(valid_dataloaders)):
                images = torch.cat((item['mri'], item['pet']), dim=0)
                labels = torch.cat((torch.zeros(int(images.shape[0] // 2)).to(torch.long), item['label']), dim=0)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss_cls = criterion(outputs, labels).item()
                val_loss += loss_cls
                _, val_predicted = torch.max(outputs.data, 1)
                val_correct_sum += (labels.data == val_predicted).sum().item()
                val_simple_cnt += labels.size(0)
                y_val_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
                y_val_pred.extend(np.ravel(np.squeeze(val_predicted.cpu().detach().numpy())).tolist())
                val_prob_all.extend(outputs[:, 1].cpu().detach().numpy())
                val_label_all.extend(labels.cpu())

        train_loss = train_loss / len(train_dataloaders)
        val_loss = val_loss / len(valid_dataloaders)
        train_acc = train_correct_sum / train_simple_cnt
        val_acc = val_correct_sum / val_simple_cnt
        train_f1_score = f1_score(y_train_true, y_train_pred, average='weighted')
        val_f1_score = f1_score(y_val_true, y_val_pred, average='weighted')
        if e % print_freq == 0:
            wandb.log({"train_acc": train_acc,
                       "train_f1": train_f1_score,
                       "val_loss": val_loss,
                       "val_acc": val_acc,
                       "val_f1": val_f1_score
                       })
            print('Epochs: {}/{}...'.format(e + 1, epochs),
                  'Trian Loss:{:.3f}...'.format(train_loss),
                  'Trian Accuracy:{:.3f}...'.format(train_acc),
                  'Trian F1 Score:{:.3f}...'.format(train_f1_score),
                  'Val Loss:{:.3f}...'.format(val_loss),
                  'Val Accuracy:{:.3f}...'.format(val_acc),
                  'Val F1 Score:{:.3f}'.format(val_f1_score),
                  )
        if e % save_epoch_freq == 0:
            torch.save(model.state_dict(), expr_dir + '/{}_net.pth'.format(e))

    end = time.time()
    runing_time = end - start
    print('Training time is {:.0f}m {:.0f}s'.format(runing_time // 60, runing_time % 60))
