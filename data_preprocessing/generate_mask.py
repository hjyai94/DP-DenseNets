# %%

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import os

# %%

root_path = '/hjy/Dataset/MICCAI_BraTS_2018_Data_Training/'


# %%

def generate_mask(path):
    data_list = os.listdir(path)
    for patient in data_list:
        images0 = nib.load(path + '/' + patient + '/' + patient + '_flair.nii.gz').get_data()
        images0 = np.expand_dims(images0, axis=0)

        images1 = nib.load(path + '/' + patient + '/' + patient + '_t1.nii.gz').get_data()
        images1 = np.expand_dims(images1, axis=0)

        images2 = nib.load(path + '/' + patient + '/' + patient + '_t1ce.nii.gz').get_data()
        images2 = np.expand_dims(images2, axis=0)

        images3 = nib.load(path + '/' + patient + '/' + patient + '_t2.nii.gz').get_data()
        images3 = np.expand_dims(images3, axis=0)

        images = np.concatenate((images0, images1, images2, images3), axis=0)

        image = np.max(images, axis=0)
        image = np.swapaxes(image, 0, 2)
        image[image > 0] = 1
        image = image.astype(dtype=np.uint8)
        image = sitk.GetImageFromArray(image)
        sitk.WriteImage(image, path + '/' + patient + '/' + patient + "_mask.nii.gz")
        print(patient)


# %%

generate_mask(root_path + 'HGG')
generate_mask(root_path + 'LGG')

# %%


