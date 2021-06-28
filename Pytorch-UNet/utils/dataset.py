import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt


class PathmlSegmentationDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, ids_to_use=False, mask_suffix='', transform=None):#A.Compose([A.Normalize(), ToTensorV2(),])):
        """transform must be a composition of Albumentation transformations
            and must include ToTensorV2() as a minimum.
        """
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mask_suffix = mask_suffix
        self.ids = []
        self.transform=transform

        #self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        #self.ids = [Path(file).stem for file in glob.glob(os.path.join(imgs_dir, '*', '*', '*.*'))]
        for file in glob(os.path.join(imgs_dir, '*', '*', '*.*')):
            if ids_to_use:
                #print(os.path.basename(os.path.dirname(os.path.dirname(file))))
                if os.path.basename(os.path.dirname(os.path.dirname(file))) in ids_to_use:
                    self.ids.append(Path(file).stem)
            else:
                self.ids.append(Path(file).stem)

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        #w, h = pil_img.size
        #newW, newH = int(scale * w), int(scale * h)
        #assert newW > 0 and newH > 0, 'Scale is too small'
        #pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        #img_trans = img_nd.transpose((2, 0, 1))
        #if img_trans.max() > 1:
        #    img_trans = img_trans / 255

        return img_nd

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(os.path.join(self.masks_dir, '*', '*', idx + self.mask_suffix + '.*'))
        #print("mask file path: "+os.path.join(self.masks_dir, '*', '*', idx + self.mask_suffix + '.*'))
        img_file = glob(os.path.join(self.imgs_dir, '*', '*', idx + '.*'))
        #print("image file path: "+os.path.join(self.imgs_dir, '*', '*', idx + '.*'))

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img)
        mask = self.preprocess(mask)
        # albumentation's normalize sadly doesn't divide mask values by 255
        # so we must do it ourselves
        #if mask.max() > 1:
        #    mask = mask / 255
        #print("img shape:", img.shape)
        #print("mask shape:", mask.shape)

        # Perform augmentation. Albumentation will apply the appropriate
        # augmentation to both img and mask in the same way, and others
        # just to the image (like color jitter)
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        #print("transformed img shape:", transformed['image'].shape)
        #print("transformed mask shape:", transformed['mask'].permute(2,0,1).shape)
        #print("transformed img max:", torch.max(transformed['image']))
        ###print("transformed mask max:", torch.max(transformed['mask'].permute(2,0,1)))

        # HWC to CHW
        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        if img.max() > 1:
            img = img / 255
        if mask.max() > 1:
            mask = mask / 255

        #img = torch.from_numpy(img).type(torch.FloatTensor)
        #mask = torch.from_numpy(mask).type(torch.FloatTensor)
        #img = img.permute(2,0,1)
        #mask = mask.permute(2,0,1)


        #print("img", img)
        #print("mask", mask)

        #img = np.asarray(img)
        #mask = np.asarray(mask)

        # Transpose images because Pytorch wants CHW images and we currently
        # have HWC images
        #img = img.transpose((2, 0, 1))
        #mask = mask.transpose((2, 0, 1))

        return {
            #'image': img,
            #'mask': mask
            #'image': transformed['image'],
            #'mask': transformed['mask'].permute(2,0,1)
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }



# adapted from https://albumentations.ai/docs/examples/example_kaggle_salt/
def visualizeSegmentationAugmentation(path_to_image, path_to_mask, transform, folder=None):
    img = Image.open(path_to_image)
    mask = Image.open(path_to_mask)
    img_nd = np.array(img)
    mask_nd = np.array(mask)

    #if transform is not None:
    transformed = transform(image=img_nd, mask=mask_nd)

    fontsize = 18

    f, ax = plt.subplots(2, 2, figsize=(8, 8))

    ax[0, 0].imshow(img_nd)
    ax[0, 0].set_title('Original image', fontsize=fontsize)

    ax[1, 0].imshow(mask_nd)
    ax[1, 0].set_title('Original mask', fontsize=fontsize)

    ax[0, 1].imshow(transformed['image'])
    ax[0, 1].set_title('Transformed image', fontsize=fontsize)

    ax[1, 1].imshow(transformed['mask'])
    ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

    if folder:
        plt.savefig(os.path.join(folder, Path(path_to_image).stem+'_augmentationexample.jpg'))


    #print("transformed image max:", torch.max(transformed['image']))
    #print("transformed mask max:", torch.max(transformed['mask'].permute(2,0,1)))









class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, ids_to_use=False, scale=1, mask_suffix='', transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = []
        self.transform=transform
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        #self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        #self.ids = [Path(file).stem for file in glob.glob(os.path.join(imgs_dir, '*', '*', '*.*'))]
        for file in glob(os.path.join(imgs_dir, '*', '*', '*.*')):
            if ids_to_use:
                #print(os.path.basename(os.path.dirname(os.path.dirname(file))))
                if os.path.basename(os.path.dirname(os.path.dirname(file))) in ids_to_use:
                    self.ids.append(Path(file).stem)
            else:
                self.ids.append(Path(file).stem)

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(os.path.join(self.masks_dir, '*', '*', idx + self.mask_suffix + '.*'))
        #print("mask file path: "+os.path.join(self.masks_dir, '*', '*', idx + self.mask_suffix + '.*'))
        img_file = glob(os.path.join(self.imgs_dir, '*', '*', idx + '.*'))
        #print("image file path: "+os.path.join(self.imgs_dir, '*', '*', idx + '.*'))

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


#class CarvanaDataset(BasicDataset):
#    def __init__(self, imgs_dir, masks_dir, scale=1):
#        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
