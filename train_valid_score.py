import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.models import ModelBuilder
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from data_loader.dataset import TrainDataset, ValidDataset, ValDataset, norm
import os
import nibabel as nib
import argparse
from utils.util import AverageMeter
from distutils.version import LooseVersion
import math
from tensorboardX import SummaryWriter
import json
from model.losses import *
import setproctitle  # pip install setproctitle
import pandas as pd
from valid import segment, test


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter()

    model.train()
    for iteration, sample in enumerate(train_loader):
        image = sample['images'].float()
        target = sample['labels'].long()
        image = Variable(image).cuda()
        label = Variable(target).cuda()

        # The dimension of out should be in the dimension of B,C,W,H,D
        # transform the prediction and label
        if args.id == 'FullyHierarchical' or args.id == 'DenseVoxNet':
            out_bu, out_td = model(image)
        elif args.id == 'Unet'  or args.id == 'DeepMedic':
            out = model(image)
        # extract the center part of the labels and output as the final prediction
        start_index = []
        end_index = []
        for i in range(3):
            start = int((args.crop_size[i] - args.center_size[i]) / 2)
            start_index.append(start)
            end_index.append(start + args.center_size[i])
        label = label[:, start_index[0]:end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]]

        if args.id == 'FullyHierarchical' or args.id == 'DenseVoxNet':
            out_bu = out_bu[:, :, start_index[0]:end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]]
            out_td = out_td[:, :, start_index[0]:end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]]
            out_bu = out_bu.permute(0, 2, 3, 4, 1).contiguous().view(-1, 5)
            out_td = out_td.permute(0, 2, 3, 4, 1).contiguous().view(-1, args.num_classes)
            label = label.contiguous().view(-1).cuda()
            loss = criterion(out_bu, label) + criterion(out_td, label)
        elif args.id == 'Unet' or args.id == 'MultiScaleUnet':
            out = out[:, :, start_index[0]:end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]]
            out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, 5)
            label = label.contiguous().view(-1).cuda()
            loss = criterion(out, label)
        elif args.id == 'DeepMedic':
            out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, args.num_classes)
            label = label.contiguous().view(-1).cuda()
            loss = criterion(out, label)
        else:
            raise Exception('Network undefined')

        # losses.update(loss.data[0],image.size(0))
        losses.update(loss.item(), image.size(0))  # changed by hjy

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # adjust learning rate
        cur_iter = iteration + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizer, cur_iter, args)

        print('   * i {} |  lr: {:.6f} | Training Loss: {losses.avg:.3f}'.format(iteration, args.running_lr,
                                                                                 losses=losses))

    print('   * EPOCH {epoch} | Training Loss: {losses.avg:.3f}'.format(epoch=epoch, losses=losses))

    return losses.avg

# validation in the validation dataset to compute each score for each epoch
def valid_score(model, epoch, args):
    model.eval()
    # initialization
    num_ignore = 0
    margin = [args.crop_size[k] - args.center_size[k] for k in range(3)]
    num_images = int(len(valid_dir))
    dice_score = np.zeros([num_images, 3]).astype(float)

    for i in range(num_images):
        # load the images, label and mask
        direct, _ = valid_dir[i].split("\n")
        _, patient_ID = direct.split('/')

        if args.correction:
            flair = nib.load(os.path.join(args.root_path, direct, patient_ID + '_flair_corrected.nii.gz')).get_data()
            t2 = nib.load(os.path.join(args.root_path, direct, patient_ID + '_t2_corrected.nii.gz')).get_data()
            t1 = nib.load(os.path.join(args.root_path, direct, patient_ID + '_t1_corrected.nii.gz')).get_data()
            t1ce = nib.load(os.path.join(args.root_path, direct, patient_ID + '_t1ce_corrected.nii.gz')).get_data()
            # print('Using bias correction dataset')
        else:
            flair = nib.load(os.path.join(args.root_path, direct, patient_ID + '_flair.nii.gz')).get_data()

            t2 = nib.load(os.path.join(args.root_path, direct, patient_ID + '_t2.nii.gz')).get_data()

            t1 = nib.load(os.path.join(args.root_path, direct, patient_ID + '_t1.nii.gz')).get_data()

            t1ce = nib.load(os.path.join(args.root_path, direct, patient_ID + '_t1ce.nii.gz')).get_data()
            # print("not using bias correction correction dataset")

        mask = nib.load(os.path.join(args.root_path, direct, patient_ID + '_mask.nii.gz')).get_data()
        labels = nib.load(os.path.join(args.root_path, direct, patient_ID + '_seg.nii.gz')).get_data()
        mask = mask.astype(int)
        labels = labels.astype(int)
        flair = np.expand_dims(norm(flair), axis=0).astype(float)
        t2 = np.expand_dims(norm(t2), axis=0).astype(float)
        t1 = np.expand_dims(norm(t1), axis=0).astype(float)
        t1ce = np.expand_dims(norm(t1ce), axis=0).astype(float)
        images = np.concatenate([flair, t2, t1, t1ce], axis=0).astype(float)

        # divide the input images input small image segments
        # return the padding input images which can be divided exactly
        image_pad, mask_pad, label_pad, num_segments, padding_index, index = segment(images, mask, labels, args)

        # initialize prediction for the whole image as background
        labels_shape = list(labels.shape)
        labels_shape.append(args.num_classes)
        pred = np.zeros(labels_shape)
        pred[:, :, :, 0] = 1

        # initialize the prediction for a small segmentation as background
        pad_shape = [int(num_segments[k] * args.center_size[k]) for k in range(3)]
        pad_shape.append(args.num_classes)
        pred_pad = np.zeros(pad_shape)
        pred_pad[:, :, :, 0] = 1

        # score_per_image stores the sum of each image
        score_per_image = np.zeros([3, 3])
        # iterate over the z dimension
        for idz in range(num_segments[2]):
            tf = ValDataset(image_pad, label_pad, mask_pad, num_segments, idz, args)
            valid_loader = DataLoader(tf, batch_size=1, shuffle=args.shuffle, num_workers=args.num_workers,
                                     pin_memory=False)
            score_seg, pred_seg = test(valid_loader, model, num_segments, args)
            pred_pad[:, :, idz * args.center_size[2]:(idz + 1) * args.center_size[2], :] = pred_seg
            score_per_image += score_seg

        # decide the start and end point in the original image
        for k in range(3):
            if index[0][k] == 0:
                index[0][k] = int(margin[k] / 2 - padding_index[0][k])
            else:
                index[0][k] = int(margin[k] / 2 + index[0][k])

            index[1][k] = int(min(index[0][k] + num_segments[k] * args.center_size[k], labels.shape[k]))

        dist = [index[1][k] - index[0][k] for k in range(3)]
        pred[index[0][0]:index[1][0], index[0][1]:index[1][1], index[0][2]:index[1][2]] = pred_pad[:dist[0], :dist[1],
                                                                                          :dist[2]]

            # compute the Enhance, Core and Whole dice score
        dice_score_per = [
            2 * np.sum(score_per_image[k, 2]) / (np.sum(score_per_image[k, 0]) + np.sum(score_per_image[k, 1])) for k in
            range(3)]
        print('Image: %d, Enhance score: %.4f, Core score: %.4f, Whole score: %.4f' % (
        i, dice_score_per[0], dice_score_per[1], dice_score_per[2]))

        dice_score[i, :] = dice_score_per

    mean_dice = np.mean(dice_score, axis=0)

    print('Enhance score: %.4f, Core score: %.4f, Whole score: %.4f, Mean Dice score: %.4f' % (
    mean_dice[0], mean_dice[1], mean_dice[2], np.mean(mean_dice)))

    return mean_dice


def save_checkpoint(state, epoch, args):
    filename = args.ckpt + '/' + str(epoch) + '_checkpoint.pth.tar'
    print(filename)
    torch.save(state, filename)


def adjust_learning_rate(optimizer, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr


def main(args):

    # import network architecture
    builder = ModelBuilder()
    model = builder.build_net(arch=args.id)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus))).cuda()
    cudnn.benchmark = True

    # collect the number of parameters in the network
    print("------------------------------------------")
    print("Network Architecture of Model %s:" % (args.id))
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul

    # print(model)
    print("Number of trainable parameters %d in Model %s" % (num_para, args.id))
    print("------------------------------------------")

    # set the optimizer and loss
    if args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), args.lr, alpha=args.alpha, eps=args.eps,
                              weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=0.0005)

    if args.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'dice_loss':
        criterion = bratsDiceLossOriginal5
    elif args.loss == 'cross_entropy_dice_loss':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'generalized_dice_loss':
        criterion = GeneralizedDiceLoss(classes=5, sigmoid_normalization=False)
    else:
        raise Exception('Loss function undefined')


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    # loading training data
    tf_train = TrainDataset(train_dir, args)
    train_loader = DataLoader(tf_train, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers,
                              pin_memory=True)

    if args.tensorboard:
        writer = SummaryWriter()
    print("Start training ...")
    avg_dice_mean_flag = False
    early_stopping = False
    avg_dice_mean = []
    for epoch in range(args.start_epoch + 1, args.num_epochs + 1):
        setproctitle.setproctitle("Epoch:{}/{}".format(epoch,args.num_epochs))
        train_loss_avg = train(train_loader, model, criterion, optimizer, epoch, args)
        if args.tensorboard:
            writer.add_scalar('train_loss', train_loss_avg, epoch)

        if epoch % 50 == 0: # validation every 50 epoch
            with torch.no_grad():
                mean_dice = valid_score(model, epoch, args)
                avg_dice_mean.append(np.mean(mean_dice))
                torch.cuda.empty_cache()
                if np.mean(mean_dice) >= np.mean(avg_dice_mean):
                    early_stopping = False
                    avg_dice_mean_flag = True
                else:
                    early_stopping = True
                    avg_dice_mean_flag = False


                if args.tensorboard:
                    writer.add_scalars('scores', {'Enhance Score': mean_dice[0],
                                                         'Core Score': mean_dice[1],'Whole Score': mean_dice[2]}, epoch)

        # save models
        if epoch > args.particular_epoch:
            if epoch % args.save_epochs_steps == 0: # or avg_dice_mean_flag == True
                save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()},
                                epoch, args)

        if early_stopping and False:
            print("Early stopping, current epoch is: ", epoch)
            break

    # export scalar data to JSON for external processing
    if args.tensorboard:
        writer.export_scalars_to_json("./runs/all_scalars.json")
        writer.close()

    print("Training Done")

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.3.0'), \
        'PyTorch>=0.3.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='FullyHierarchical',
                        help='a name for identitying the model.  Choose from FullyHierarchical, DenseVoxNet, Unet, DeepMedic')
    parser.add_argument('--loss', default='cross_entropy',
                        help='a name of loss function. Choose from the following options: cross_entropy, dice_loss, cross_entropy_dice_loss.')

    # Path related arguments
    parser.add_argument('--train_path', default='datalist/train.txt',
                        help='text file of the name of training data')
    parser.add_argument('--valid_path', default='datalist/valid.txt',
                        help='text file of the name of validation data')
    parser.add_argument('--root_path', default='/hjy/Dataset/MICCAI_BraTS_2018_Data_Training',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved/models',
                        help='folder to output checkpoints')

    # Data related arguments
    parser.add_argument('--crop_size', default=[38, 38, 38], nargs='+', type=int,
                        help='crop size of the input image (int or list)')
    parser.add_argument('--center_size', default=[20, 20, 20], nargs='+', type=int,
                        help='the corresponding output size of the input image (int or list)')
    parser.add_argument('--num_classes', default=5, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=5, type=int,
                        help='number of input image for each patient include four modalities and the mask')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--random_flip', default=True, type=bool,
                        help='if horizontally flip images when training')
    parser.add_argument('--random_augment', default=False, type=bool,
                        help='randomly augment data')
    parser.add_argument('--normalization', default=True, type=bool,
                        help='normalizae the data before training')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='if shuffle the data during training')
    parser.add_argument('--mask', default=True, type=bool,
                        help='if have the mask')

    # optimization related arguments
    parser.add_argument('--num_gpus', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=3, type=int,
                        help='training batch size')
    parser.add_argument('--num_epochs', default=1000, type=int,
                        help='epochs for training')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='start learning rate')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop learning rate')
    parser.add_argument('--optim', default='RMSprop', help='optimizer')
    parser.add_argument('--alpha', default='0.9', type=float, help='alpha in RMSprop')
    parser.add_argument('--eps', default=10 ** (-4), type=float, help='eps in RMSprop')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weights regularizer')
    parser.add_argument('--momentum', default=0.6, type=float, help='momentum for RMSprop')
    parser.add_argument('--save_epochs_steps', default=10, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--particular_epoch', default=200, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')
    parser.add_argument('--num_round', default=1, type=int)
    parser.add_argument('--correction', type=bool, default=False)
    parser.add_argument('--tensorboard', action='store_true', help='save tensorboard file')

    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    #  open training file
    train_file = open(args.train_path, 'r')
    train_dir = train_file.readlines()
    # open validation file
    valid_file = open(args.valid_path, 'r')
    valid_dir = valid_file.readlines()

    print('numbers of patient Id', len(train_dir))
    args.ckpt = os.path.join(args.ckpt, args.id, str(args.num_round))
    print('Models are saved at %s' % (args.ckpt))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    if args.start_epoch > 1:
        args.resume = args.ckpt + '/' + str(args.start_epoch) + '_checkpoint.pth.tar'

    args.running_lr = args.lr
    args.epoch_iters = math.ceil(int(len(train_dir)) / args.batch_size)
    args.max_iters = args.epoch_iters * args.num_epochs


    assert isinstance(args.crop_size, (int, list))
    if len(args.crop_size) == 1:
        args.crop_size = [*args.crop_size, *args.crop_size, *args.crop_size]

    assert isinstance(args.center_size, (int, list))
    if len(args.center_size) == 1:
        args.center_size = [*args.center_size, *args.center_size, *args.center_size]

    # save arguments to config file
    with open(args.ckpt + '/' + args.id + '.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    main(args)

