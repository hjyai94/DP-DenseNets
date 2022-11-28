from torch.utils.data import Dataset
import numpy as np
import random
import nibabel as nib
import os
import cv2

def norm(image):
    image = np.squeeze(image)
    image_nonzero = image[np.nonzero(image)]
    return (image - image_nonzero.mean()) / image_nonzero.std()

class TrainDataset(Dataset):
    def __init__(self, root_dir, args):   
        self.root_dir = root_dir
        self.num_input = args.num_input
        self.length = int(len(self.root_dir))
        self.crop_size = args.crop_size
        self.random_flip = args.random_flip
        self.random_augment = args.random_augment
        self.root_path = args.root_path
        self.correction = args.correction
        assert args.mask, "Missing mask as the input"
        assert args.normalization, "You need to do the data normalization before training"
    
    def __len__(self):
        return self.length
   
    def __getitem__(self, idx):
        direct, _ = self.root_dir[idx].split("\n")
        _, patient_ID = direct.split('/')

        if self.correction:
            flair = nib.load(os.path.join(self.root_path, direct, patient_ID + '_flair_corrected.nii.gz')).get_data()
            t2 = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t2_corrected.nii.gz')).get_data()
            t1 = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t1_corrected.nii.gz')).get_data()
            t1ce = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t1ce_corrected.nii.gz')).get_data()
            # print('Using bias correction dataset')
        else:
            flair = nib.load(os.path.join(self.root_path, direct, patient_ID + '_flair.nii.gz')).get_data()

            t2 = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t2.nii.gz')).get_data()

            t1 = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t1.nii.gz')).get_data()

            t1ce = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t1ce.nii.gz')).get_data()
            # print("not using bias correction correction dataset")

        mask = nib.load(os.path.join(self.root_path, direct, patient_ID + '_mask.nii.gz')).get_data()
        labels = nib.load(os.path.join(self.root_path, direct, patient_ID + '_seg.nii.gz')).get_data()
        mask = mask.astype(int)
        labels = labels.astype(int)
        flair = np.expand_dims(norm(flair), axis=0).astype(float)
        t2 = np.expand_dims(norm(t2), axis=0).astype(float)
        t1 = np.expand_dims(norm(t1), axis=0).astype(float)
        t1ce = np.expand_dims(norm(t1ce), axis=0).astype(float)
        images = np.concatenate([flair, t2, t1, t1ce], axis=0).astype(float)

        # print("images shape", images.shape, "labels shape", labels.shape, "mask shape", mask.shape)
        # images shape: 4 x H x W x D 
        # labels shape: H x W x D 
        sample = {'images': images, 'mask': mask, 'labels':labels}
        transform = RandomCrop(self.crop_size, self.random_flip, self.num_input, self.random_augment)
        sample = transform(sample)
        return sample


#validation for computing loss for each epoch
class ValidDataset(Dataset):
    def __init__(self, root_dir, args):
        self.root_dir = root_dir
        self.num_input = args.num_input
        self.length = int(len(self.root_dir))
        self.crop_size = args.crop_size
        self.random_flip = args.random_flip
        self.random_augment = args.random_augment
        self.root_path = args.root_path
        self.correction = args.correction
        assert args.mask, "Missing mask as the input"
        assert args.normalization, "You need to do the data normalization before training"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        direct, _ = self.root_dir[idx].split("\n")
        _, patient_ID = direct.split('/')

        if self.correction:
            flair = nib.load(os.path.join(self.root_path, direct, patient_ID + '_flair_corrected.nii.gz')).get_data()
            t2 = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t2_corrected.nii.gz')).get_data()
            t1 = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t1_corrected.nii.gz')).get_data()
            t1ce = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t1ce_corrected.nii.gz')).get_data()
            # print('Using bias correction dataset')
        else:
            flair = nib.load(os.path.join(self.root_path, direct, patient_ID + '_flair.nii.gz')).get_data()

            t2 = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t2.nii.gz')).get_data()

            t1 = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t1.nii.gz')).get_data()

            t1ce = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t1ce.nii.gz')).get_data()
            # print("not using bias correction correction dataset")

        mask = nib.load(os.path.join(self.root_path, direct, patient_ID + '_mask.nii.gz')).get_data()
        labels = nib.load(os.path.join(self.root_path, direct, patient_ID + '_seg.nii.gz')).get_data()
        mask = mask.astype(int)
        labels = labels.astype(int)
        flair = np.expand_dims(norm(flair), axis=0).astype(float)
        t2 = np.expand_dims(norm(t2), axis=0).astype(float)
        t1 = np.expand_dims(norm(t1), axis=0).astype(float)
        t1ce = np.expand_dims(norm(t1ce), axis=0).astype(float)
        images = np.concatenate([flair, t2, t1, t1ce], axis=0).astype(float)

        # print("images shape", images.shape, "labels shape", labels.shape, "mask shape", mask.shape)
        # images shape: 4 x H x W x D
        # labels shape: H x W x D
        sample = {'images': images, 'mask': mask, 'labels': labels}
        transform = RandomCrop(self.crop_size, self.random_flip, self.num_input, self.random_augment)
        sample = transform(sample)
        return sample

# validation for computing each score
class ValDataset(Dataset):
    def __init__(self, image, label, mask, num_segments, idz, args):
        self.images = image
        self.labels = label
        self.mask = mask
        self.numx = num_segments[0]
        self.numy = num_segments[1]
        self.idz = idz
        self.center_size = args.center_size
        self.crop_size = args.crop_size
        self.num_input = args.num_input - 1

    def __len__(self):
        return self.numy

    def __getitem__(self, idy):
        h, w, d = self.crop_size
        left = np.arange(self.numx) * self.center_size[0]
        bottom = idy * self.center_size[1]
        forward = self.idz * self.center_size[2]

        image = np.zeros([self.numx, self.num_input, h, w, d])
        mask = np.zeros([self.numx, h, w, d])
        label = np.zeros([self.numx, h, w, d])

        # dimension of label and image B x H x W x D
        for i in range(self.numx):
            image[i, :] = self.images[:, left[i]: left[i] + h, bottom: bottom + w, forward: forward + d]
            label[i, :] = self.labels[left[i]: left[i] + h, bottom: bottom + w, forward: forward + d]
            mask[i, :] = self.mask[left[i]: left[i] + h, bottom: bottom + w, forward: forward + d]

        # images shape: H x W x D
        # labels shape: H x W x D
        sample = {'images': image, 'labels': label, 'mask': mask}
        return sample


class ValDataset_full(Dataset):
    def __init__(self, root_dir, args):
        self.root_dir = root_dir
        self.num_input = args.num_input
        self.length = int(len(self.root_dir))
        self.crop_size = args.crop_size
        self.random_flip = args.random_flip
        self.random_augment = args.random_augment
        self.root_path = args.root_path
        self.correction = args.correction
        assert args.mask, "Missing mask as the input"
        assert args.normalization, "You need to do the data normalization before training"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        direct, _ = self.root_dir[idx].split("\n")
        _, patient_ID = direct.split('/')

        if self.correction:
            flair = nib.load(os.path.join(self.root_path, direct, patient_ID + '_flair_corrected.nii.gz')).get_data()
            t2 = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t2_corrected.nii.gz')).get_data()
            t1 = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t1_corrected.nii.gz')).get_data()
            t1ce = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t1ce_corrected.nii.gz')).get_data()
            # print('Using bias correction dataset')
        else:
            flair = nib.load(os.path.join(self.root_path, direct, patient_ID + '_flair.nii.gz')).get_data()

            t2 = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t2.nii.gz')).get_data()

            t1 = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t1.nii.gz')).get_data()

            t1ce = nib.load(os.path.join(self.root_path, direct, patient_ID + '_t1ce.nii.gz')).get_data()
            # print("not using bias correction correction dataset")

        mask = nib.load(os.path.join(self.root_path, direct, patient_ID + '_mask.nii.gz')).get_data()
        labels = nib.load(os.path.join(self.root_path, direct, patient_ID + '_seg.nii.gz')).get_data()
        mask = mask.astype(int)
        labels = labels.astype(int)
        flair = np.expand_dims(norm(flair), axis=0).astype(float)
        t2 = np.expand_dims(norm(t2), axis=0).astype(float)
        t1 = np.expand_dims(norm(t1), axis=0).astype(float)
        t1ce = np.expand_dims(norm(t1ce), axis=0).astype(float)
        images = np.concatenate([flair, t2, t1, t1ce], axis=0).astype(float)

        # print("images shape", images.shape, "labels shape", labels.shape, "mask shape", mask.shape)
        # images shape: 4 x H x W x D
        # labels shape: H x W x D
        sample = {'images': images, 'mask': mask, 'labels': labels}
        return sample

class TestDataset(Dataset):
    def __init__(self, image, mask, num_segments, idz, args):
        self.images = image
        self.mask = mask
        self.numx = num_segments[0]
        self.numy = num_segments[1]
        self.idz = idz
        self.center_size = args.center_size
        self.crop_size = args.crop_size
        self.num_input = args.num_input

    def __len__(self):
        return self.numy

    def __getitem__(self, idy):
        h, w, d = self.crop_size
        left =  np.arange(self.numx) * self.center_size[0]
        bottom =  idy * self.center_size[1]
        forward = self.idz * self.center_size[2]

        image = np.zeros([self.numx, self.num_input, h, w, d])
        mask = np.zeros([self. numx, h, w, d])

        # dimension of label and image B x H x W x D
        for i in range(self.numx):
            image[i,:] = self.images[:, left[i]: left[i] + h, bottom : bottom + w, forward : forward + d]
            mask[i,:] = self.mask[left[i]: left[i] + h, bottom : bottom + w, forward : forward + d]

        # images shape: H x W x D
        # labels shape: H x W x D
        sample = {'images': image, 'mask':mask}
        return sample

def resize_image(im, size, interp=cv2.INTER_LINEAR):
    im_resized = cv2.resize(im, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
    #add last dimension again if it was removed by resize
    if im.ndim > im_resized.ndim:
        im_resized = np.expand_dims(im_resized, im.ndim)
    return im_resized

def dense_image_warp(im, dx, dy, interp=cv2.INTER_LINEAR):

    map_x, map_y = deformation_to_transformation(dx, dy)

    do_optimization = (interp == cv2.INTER_LINEAR)
    # The following command converts the maps to compact fixed point representation
    # this leads to a ~20% increase in speed but could lead to accuracy losses
    # Can be uncommented
    if do_optimization:
        map_x, map_y = cv2.convertMaps(map_x, map_y, dstmap1type=cv2.CV_16SC2)

    remapped = cv2.remap(im, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REFLECT) #borderValue=float(np.min(im)))
    if im.ndim > remapped.ndim:
        remapped = np.expand_dims(remapped, im.ndim)
    return remapped

def rotate_image(img, angle, interp=cv2.INTER_LINEAR):
    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    out = cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp, borderMode=cv2.BORDER_REPLICATE)
    return np.reshape(out, img.shape)

def cropToSize(image, targetSize):
    offsetX = (image.shape[0] - targetSize[0]) // 2
    endX = offsetX + targetSize[0]
    offsetY = (image.shape[1] - targetSize[1]) // 2
    endY = offsetY + targetSize[1]
    return image[offsetX:endX, offsetY:endY, :]

def padToSize(image, targetSize, backgroundColor):
    offsetX = (targetSize[0] - image.shape[0]) // 2
    endX = offsetX + image.shape[0]
    offsetY = (targetSize[1] - image.shape[1]) // 2
    endY = offsetY + image.shape[1]
    targetSize.append(image.shape[2]) #add channels to shape
    paddedImg = np.ones(targetSize, dtype=np.float32) * backgroundColor
    paddedImg[offsetX:endX, offsetY:endY, :] = image
    return paddedImg

def convertToOnehot(labels):
    shape = labels.shape
    out = np.zeros([shape[0], shape[1], shape[2], 5])
    for i in range(5):
        out[:, :, :, i] = (labels == i)

    return out

def deformation_to_transformation(dx, dy):

    nx, ny = dx.shape

    grid_y, grid_x = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)

    return map_x, map_y

class RandomCrop(object):
    def __init__(self, output_size, random_flip, num_input, random_augment):
        assert len(output_size) == 3
        self.output_size = output_size        
        self.random_flip = random_flip
        self.num_input = num_input
        self.random_augment = random_augment
    
    def __call__(self, sample):
        images, labels, mask = sample['images'], sample['labels'], sample['mask']   
        h, w, d = self.output_size
       
        # generate the training batch with equal probability for the foreground and background
        # within the mask
        labelm = labels + mask
        # choose foreground or background
        fb = np.random.choice(2)
        if fb:
            index = np.argwhere(labelm > 1)
        else:
            index = np.argwhere(labelm == 1)
        # choose the center position of the image segments
        choose = random.sample(range(0, len(index)), 1)
        center = index[choose].astype(int)
        center = center[0]
        
        # check whether the left and right index overflow
        left = []
        for i in range(3):
        	margin_left = int(self.output_size[i]/2)
        	margin_right = self.output_size[i] - margin_left
        	left_index = center[i] - margin_left
        	right_index = center[i] + margin_right
        	if left_index < 0:
        		left_index = 0
        	if right_index > labels.shape[i]:
        		left_index = left_index - (right_index - labels.shape[i])
        	left.append(left_index)
        	
        # crop the image and the label to generate image segments
        image = np.zeros([self.num_input - 1, h, w, d])
        label = np.zeros([h, w, d])
        
        image = images[:, left[0]:left[0] + h, left[1]:left[1] + w, left[2]:left[2] + d]
        label = labels[left[0]:left[0] + h, left[1]:left[1] + w, left[2]:left[2] + d]        

        image = image.transpose((1, 2, 3, 0)) ##from 4xHxWxD to HxWxDx4(HxWxDx4)
        label = convertToOnehot(label) ## HxWxDx5

        xSize = image.shape[0]
        ySize = image.shape[1]
        zSize = image.shape[2]
        defaultPerChannel = image[0, 0, 0, :]
        defaultLabelValues = np.asarray([1, 0, 0, 0, 0], dtype=np.float32)

        # random flip
        if self.random_flip:       
        	flip = np.random.choice(2)*2-1
        	image = image[:,:,::flip,:]
        	label = label[:,:,::flip,:]


        # # RANDOM SCALE
        if self.random_augment:
            scaleFactor = 1.1
            scale = np.random.uniform(1 / scaleFactor, 1 * scaleFactor)
            for z in range(zSize):
                scaledSize = [round(xSize * scale), round(ySize * scale)]
                imgScaled = resize_image(image[:, :, z, :], scaledSize)
                lblScaled = resize_image(label[:, :, z, :], scaledSize, cv2.INTER_NEAREST)
                if scale < 1:
                    image[:, :, z, :] = padToSize(imgScaled, [xSize, ySize], defaultPerChannel)
                    label[:, :, z, :] = padToSize(lblScaled, [xSize, ySize], defaultLabelValues)
                else:
                    image[:, :, z, :] = cropToSize(imgScaled, [xSize, ySize])
                    label[:, :, z, :] = cropToSize(lblScaled, [xSize, ySize])

        # random intensity shift
        if self.random_augment:
            MAX_INTENSITY_SHIFT = 0.1
            for i in range(4):  # number of channels
                image[:, :, :, i] = image[:, :, :, i] + np.random.uniform(-MAX_INTENSITY_SHIFT, MAX_INTENSITY_SHIFT)  # assumes unit std derivation

        # ROTATE
        if self.random_augment:
            rotDegrees = 20
            random_angle = np.random.uniform(-rotDegrees, rotDegrees)
            for z in range(zSize):
                image[:,:,z,:] = rotate_image(image[:, :, z, :], random_angle)
                label[:,:,z,:] = rotate_image(label[:, :, z, :], random_angle, cv2.INTER_NEAREST)

        # RANDOM ELASTIC DEFOMRATIONS (like in U-NET)
        if self.random_augment:
            mu = 0
            sigma = 10
            dx = np.random.normal(mu, sigma, 9)
            dx_mat = np.reshape(dx, (3, 3))
            dx_img = resize_image(dx_mat, (xSize, ySize), interp=cv2.INTER_CUBIC)

            dy = np.random.normal(mu, sigma, 9)
            dy_mat = np.reshape(dy, (3, 3))
            dy_img = resize_image(dy_mat, (xSize, ySize), interp=cv2.INTER_CUBIC)

            for z in range(zSize):
                image[:, :, z, :] = dense_image_warp(image[:, :, z, :], dx_img, dy_img)
                label[:, :, z, :] = dense_image_warp(label[:, :, z, :], dx_img, dy_img, cv2.INTER_NEAREST)

        image = image.transpose((3, 0, 1, 2)) ##from HxWxDxC to CxHxWxD
        label = np.argmax(label, axis=-1)  ## HxWxD
        return {'images':image.copy(), 'labels': label.copy()}

