import torch
import torch.nn as nn
import numpy as np
from model.models import ModelBuilder
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import time
import random
import nibabel as nib
from utils import AverageMeter
from distutils.version import LooseVersion
import argparse
from data_loader.dataset import ValDataset, norm
import SimpleITK as sitk
import logging
import cv2


# save prediction results in the format of online submission
def visualize_result(name, pred, args):
    pred = nib.Nifti1Image(pred, None)
    nib.save(pred, args.result + '/' + str(args.num_round) + '/' + str(name) + '.nii.gz')


# compute the number of segments of  the test images
def segment(image, mask, label, args):
    # find the left, right, bottom, top, forward, backward limit of the mask   
    boundary = np.nonzero(mask)
    boundary = [np.unique(boundary[i]) for i in range(3)]
    limit = [(min(boundary[i]), max(boundary[i])) for i in range(3)]
        
    # compute the number of image segments in an image 
    num_segments = [int(np.ceil((limit[i][1] - limit[i][0]) / args.center_size[i])) for i in range(3)]   

    # compute the margin and the shape of the padding image 
    margin = [args.crop_size[i] - args.center_size[i] for i in range(3)]
    padding_image_shape = [num_segments[i] * args.center_size[i] + margin[i] for i in range(3)]  
    
    # start_index is corresponding to the location in the new padding images
    start_index = [limit[i][0] - int(margin[i]/2) for i in range(3)]
    start_index = [int(-index) if index < 0 else 0 for index in start_index]
    
    # start and end is corresponding to the location in the original images
    start = [int(max(limit[i][0] - int(margin[i]/2), 0)) for i in range(3)]
    end = [int(min(start[i] + padding_image_shape[i], mask.shape[i])) for i in range(3)]

    # compute the end_index corresponding to the new padding images
    end_index =[int(start_index[i] + end[i] - start[i]) for i in range(3)]

    # initialize the padding images
    #size = [start_index[i] if start_index[i] > 0 and end[i] < mask.shape[i] else 0 for i in range(3)]
    # fix don't broadcast bug
    size = [start_index[i] if start_index[i] > 0  else 0 for i in range(3)]
    if sum(size) != 0:
        padding_image_shape = [sum(x) for x in zip(padding_image_shape, size)]

    mask_pad = np.zeros(padding_image_shape) 
    label_pad = np.zeros(padding_image_shape) 
    padding_image_shape.insert(0, args.num_input - 1)
    image_pad = np.zeros(padding_image_shape) 

    # assign the original images to the padding images
    image_pad[:, start_index[0] : end_index[0], start_index[1] : end_index[1], start_index[2] : end_index[2]] = image[:, start[0]: end[0], start[1]:end[1], start[2]:end[2]]
    label_pad[start_index[0] : end_index[0], start_index[1] : end_index[1], start_index[2] : end_index[2]] = label[start[0]: end[0], start[1]:end[1], start[2]:end[2]]
    mask_pad[start_index[0] : end_index[0], start_index[1] : end_index[1], start_index[2] : end_index[2]] = mask[start[0]: end[0], start[1]:end[1], start[2]:end[2]]
    return image_pad, mask_pad, label_pad, num_segments, (start_index, end_index), (start, end)
                
def accuracy(pred, mask, label):
    # columns in score is (# pred, # label, pred and label)
    score = np.zeros([3,5])

    # compute Enhance score (label==4) in the first line
    score[0,0] = np.count_nonzero(pred * mask == 4)
    score[0,1] = np.count_nonzero(label == 4)
    score[0,2] = np.count_nonzero(pred * mask * label == 16) # TP for enhance tumor
    score[0,3] = np.count_nonzero(pred + label == 0) + np.count_nonzero(pred * label == 1) + np.count_nonzero(pred * label == 4) # TN
    score[0,4] = np.count_nonzero(label >= 0)  # TN, TP, FP , FN

    # compute Core score (label == 1,2,4) in the second line
    pred[pred > 2] = 1
    label[label > 2] = 1
    score[1,0] = np.count_nonzero(pred * mask == 1)
    score[1,1] = np.count_nonzero(label == 1)
    score[1,2] = np.count_nonzero(pred * mask * label == 1)
    score[1,3] = np.count_nonzero(pred + label == 0) + np.count_nonzero(pred * label == 4)  # TN for core tumor
    score[1,4] = np.count_nonzero(label >= 0)  # TN, TP, FP, FN

    # compute Whole score (all labels) in the third line
    pred[pred > 1] = 1
    label[label > 1] = 1
    score[2,0] = np.count_nonzero(pred * mask == 1)
    score[2,1] = np.count_nonzero(label == 1)
    score[2,2] = np.count_nonzero(pred * mask * label == 1)
    score[2,3] = np.count_nonzero(pred + label == 0) # TN for whole tumor
    score[2,4] = np.count_nonzero(label >= 0) # TN, TP, FP , FN

    return score

def calculate_roi_unc(entropy_input):
    gaussian_kernel = (5, 5)
    image_blur = cv2.GaussianBlur(entropy_input, gaussian_kernel, sigmaX=0.1)
    image_binary = ((image_blur > 0.2)).astype('uint8')

    erosion_kernel = np.ones((5,5),np.uint8)  
    erosion = cv2.erode(entropy_input,erosion_kernel,iterations = 1)

    dilation_kernel = np.ones((5,5),np.uint8)  
    dilation = cv2.dilate(erosion,dilation_kernel,iterations = 1)

    return dilation * entropy_input

def test(test_loader, model, num_segments, args):
    # switch to evaluate mode
    model.eval()
    
    # columns in score is (# pred, # label, pred and label)
    # lines in score is (Enhance, Core, Whole)
    score = np.zeros([3, 5])
    h_c, w_c, d_c = args.center_size    
    pred_seg = np.zeros([num_segments[0] * h_c, num_segments[1] * w_c, d_c, args.num_classes])

    for i, sample in enumerate(test_loader):
        image = sample['images'].float().cuda()
        target = sample['labels'].long().cuda()
        mask = sample['mask'].long().cuda()
        
        image = torch.squeeze(image, 0)
        target = torch.squeeze(target, 0)
        mask = torch.squeeze(mask, 0)
        
        with torch.no_grad():      
            image = Variable(image)
            label = Variable(target)
            mask = Variable(mask)
        
            # The dimension of out should be in the dimension of B,C,H,W,D
            # crop center of the output
            if args.id == 'FullyHierarchical':
                _, out = model(image)
            elif args.id == 'DenseVoxNet':
                out_1, out_2 = model(image)
                out = (out_1 + out_2)// 2.0
            elif args.id == 'Unet':
                out = model(image)

            if args.id != 'DeepMedic':
                crop_center_index = (args.crop_size[0] - args.center_size[0]) // 2

                out = out[:, :, crop_center_index:-crop_center_index, crop_center_index:-crop_center_index,
                     crop_center_index:-crop_center_index]
            else:
                out = model(image)

            prob = torch.softmax(out, dim=1)
            output_entropy = - (prob * torch.log(prob + 1e-6))[:,1:].mean(1).mean(0).cpu().numpy()
            normalized_output_entropy = (output_entropy - output_entropy.min())/(output_entropy.max()-output_entropy.min()+1e-6)
            roi_unc = calculate_roi_unc(normalized_output_entropy)

            out_size = out.size()[2:]
            out = out.permute(0,2,3,4,1).contiguous().cuda()
                
            out_data = (out.data).cpu().numpy()
           
            # make the prediction
            out = out.view(-1, args.num_classes).cuda()
            prediction = torch.max(out, 1)[1].cuda().data.squeeze()
        
            # extract the center part of the label and mask
            start = [int((args.crop_size[k] - out_size[k])/2) for k in range(3)]
            end = [sum(x) for x in zip(start, out_size)]
            label = label[:, start[0]: end[0], start[1]: end[1], start[2]: end[2]]
            label = label.contiguous().view(-1)       
            mask = mask[:, start[0]: end[0], start[1]: end[1], start[2]: end[2]]
            mask = mask.contiguous().view(-1)
        
        for j in range(num_segments[0]):
            pred_seg[j*h_c:(j+1)*h_c, i*d_c: (i+1)*d_c, :, :] = out_data[j, :]    

        # compute the dice score
        score += accuracy(prediction.data.cpu().numpy(), mask.data.cpu().numpy(), label.data.cpu().numpy())
    return score, pred_seg, roi_unc.mean()

def main(args):
    # import network architecture
    builder = ModelBuilder()
    model = builder.build_net(arch=args.id)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus))).cuda()
    cudnn.benchmark = True
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)           
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            raise Exception("=> No checkpoint found at '{}'".format(args.resume))         
    
    # initialization      
    num_ignore = 0
    margin = [args.crop_size[k] - args.center_size[k] for k in range(3)]
    num_images = int(len(test_dir))
    dice_score = np.zeros([num_images, 3]).astype(float)
    TPR_score = np.zeros([num_images, 3]).astype(float)
    precision_score = np.zeros([num_images, 3]).astype(float)
    acc_score = np.zeros([num_images, 3]).astype(float)

    start_time = time.time()
    roi_result = AverageMeter()

    for i in range(num_images):
        # load the images, label and mask
        direct, _ = test_dir[i].split("\n")
        _, patient_ID = direct.split('/')

        roi_result_image = AverageMeter()

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
        pred[:,:,:,0] = 1
            
        # initialize the prediction for a small segmentation as background
        pad_shape = [int(num_segments[k] * args.center_size[k]) for k in range(3)]
        pad_shape.append(args.num_classes)
        pred_pad = np.zeros(pad_shape)  
        pred_pad[:,:,:,0] = 1 

        # score_per_image stores the sum of each image
        score_per_image = np.zeros([3, 5])


        # iterate over the z dimension
        for idz in range(num_segments[2]):
            tf = ValDataset(image_pad, label_pad, mask_pad, num_segments, idz, args)
            test_loader = DataLoader(tf, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=False)
            score_seg, pred_seg, roi_unc = test(test_loader, model, num_segments, args)
            pred_pad[:, :, idz*args.center_size[2]:(idz+1)*args.center_size[2], :] = pred_seg        
            score_per_image += score_seg

            roi_result.update(roi_unc)
            roi_result_image.update(roi_unc)
                
        # decide the start and end point in the original image
        for k in range(3):
            if index[0][k] == 0:
                index[0][k] = int(margin[k]/2 - padding_index[0][k])
            else:
                index[0][k] = int(margin[k]/2 + index[0][k])

            index[1][k] = int(min(index[0][k] + num_segments[k] * args.center_size[k], labels.shape[k]))

        dist = [index[1][k] - index[0][k] for k in range(3)]
        pred[index[0][0]:index[1][0], index[0][1]:index[1][1], index[0][2]:index[1][2]] = pred_pad[:dist[0], :dist[1], :dist[2]]
            
        if np.sum(score_per_image[0,:]) == 0 or np.sum(score_per_image[1,:]) == 0 or np.sum(score_per_image[2,:]) == 0:
            num_ignore += 1
            continue 
        # compute the Enhance, Core and Whole dice score
        dice_score_per = [2 * np.sum(score_per_image[k,2]) / (np.sum(score_per_image[k,0]) + np.sum(score_per_image[k,1])) for k in range(3)]
        TPR_score_per = [np.sum(score_per_image[k,2]) / np.sum(score_per_image[k,1]) for k in range(3)]
        precision_score_per = [np.sum(score_per_image[k,2]) / np.sum(score_per_image[k,0]) for k in range(3)]
        acc_score_per = [np.sum(score_per_image[k,3]) / np.sum(score_per_image[k,4]) for k in range(3)]


        print('Image: %d, Enhance score: %.4f, Core score: %.4f, Whole score: %.4f' % (i, dice_score_per[0], dice_score_per[1], dice_score_per[2]))           
        # acc_score_per =
        dice_score[i, :] = dice_score_per
        TPR_score[i, :] = TPR_score_per
        precision_score[i, :] = precision_score_per
        acc_score[i, :] = acc_score_per
        print('RoI Uncertainty:', roi_result_image.avg)

        if args.visualize:
            vis = np.argmax(pred, axis=3)
            vis = vis.transpose(1, 0, 2)
            visualize_result(patient_ID, vis, args)

    count_image = num_images - num_ignore
    dice_score = dice_score[:count_image,:]
    end_time = time.time()
    np.savetxt('./result/' + args.id + '/' + str(args.num_round) + '/' + args.id + '_dice.txt', dice_score)
    mean_dice = np.mean(dice_score, axis=0)
    std_dice = np.std(dice_score, axis=0)

    print('Evalution Done!')
    print('Average Inference Time', (end_time-start_time)//num_images)
    print('Enhance score: %.4f, Core score: %.4f, Whole score: %.4f, Mean Dice score: %.4f' % (mean_dice[0], mean_dice[1], mean_dice[2], np.mean(mean_dice)))
    print('Enhance TPR: %.4f, Core TPR: %.4f, Whole TPR: %.4f' % (np.mean(TPR_score, axis=0)[0], np.mean(TPR_score, axis=0)[1], np.mean(TPR_score, axis=0)[2]))
    print('Enhance precision: %.4f, Core precision: %.4f, Whole precision: %.4f' % (np.mean(precision_score, axis=0)[0], np.mean(precision_score, axis=0)[1], np.mean(precision_score, axis=0)[2]))
    # print('Enhance Accuracy: %.4f, Core accuracy: %.4f, Whole accuracy: %.4f' % (np.mean(acc_score, axis=0)[0], np.mean(acc_score, axis=0)[1], np.mean(acc_score, axis=0)[2]))
    print('Enhance std: %.4f, Core std: %.4f, Whole std: %.4f, Mean Std: %.4f' % (std_dice[0], std_dice[1], std_dice[2], np.mean(std_dice)))
    print('RoI Uncertainty:', roi_result.avg)

    logging.info("test epoch is {}".format(args.test_epoch))
    logging.info('Enhance score: %.4f, Core score: %.4f, Whole score: %.4f, Mean Dice score: %.4f' % (
        (mean_dice[0], mean_dice[1], mean_dice[2], np.mean(mean_dice))))



if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.3.0'), \
        'PyTorch>=0.3.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='FullyHierarchical',
                        help='a name for identitying the model.')

    # Path related arguments
    parser.add_argument('--test_path', default='datalist/valid_0_first_paper.txt',
                        help='txt file of the name of test data')
    parser.add_argument('--root_path', default='/home/hjy/ssd/Dataset/MICCAI_BraTS_2018_Data_Training',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved/models',
                        help='folder to output checkpoints')

    # Data related arguments
    parser.add_argument('--crop_size', default=[48,48,48], nargs='+', type=int,
                        help='crop size of the input image (int or list)')
    parser.add_argument('--center_size', default=[30,30,30], nargs='+', type=int,
                        help='the corresponding output size of the input image (int or list)')
    parser.add_argument('--num_classes', default=5, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=5, type=int,
                        help='number of input image for each patient plus the mask')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--normalization', default=True, type=bool,
                        help='normalizae the data before running the test')
    parser.add_argument('--shuffle', default=False, type=bool,
                        help='if shuffle the data in test')
    parser.add_argument('--mask', default=True, type=bool,
                        help='if have the mask')

    # test related arguments
    parser.add_argument('--num_gpus', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='test batch size')
    parser.add_argument('--test_epoch', default=400, type=int,
                        help='epoch to start test.')  
    parser.add_argument('--visualize', action='store_true',
                        help='save the prediction result as 3D images')
    parser.add_argument('--result', default='./result',
                        help='folder to output prediction results')
    parser.add_argument('--num_round', default=None, type=int,
                        help='restore the models from which run')
    parser.add_argument('--correction', dest='correction', type=bool, default=False)

    args = parser.parse_args()
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    test_file = open(args.test_path, 'r')
    test_dir = test_file.readlines()

    if not args.num_round:
        args.ckpt = os.path.join(args.ckpt, args.id)
    else:
        args.ckpt = os.path.join(args.ckpt, args.id, str(args.num_round))

    args.result = os.path.join(args.result, args.id)
    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    if not os.path.isdir(args.result + '/' + str(args.num_round)):
        os.makedirs(args.result + '/' + str(args.num_round))

    assert isinstance(args.crop_size, (int, list))
    if len(args.crop_size) == 1:
        args.crop_size = [*args.crop_size, *args.crop_size, *args.crop_size]

    assert isinstance(args.center_size, (int, list))
    if len(args.center_size) == 1:
        args.center_size = [*args.center_size, *args.center_size, *args.center_size]

    # do the test on a series of models
    args.resume = args.ckpt + '/' + str(args.test_epoch) + '_checkpoint.pth.tar'

    logging.basicConfig(filename='result/' + args.id + '/logger.log', level=logging.INFO)

    main(args)

   











