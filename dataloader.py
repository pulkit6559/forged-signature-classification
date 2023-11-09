import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image
from collections import defaultdict


class SignatureContrastiveDataset(Dataset):
    """
    Custom dataset for signature verification task.

    Args:
        data_path (str): The root directory containing genuine and forged signature images.
        
    Attributes:
        image_pairs (list): List of image pairs for signature verification.
        targets (list): List of corresponding labels (0 for genuine, 1 for forged).

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Retrieves a pair of images and their label at the specified index.
    """

    def __init__(self, data_path="None"):
        assert data_path != None
        
        self.data_path = data_path
        self.image_pairs, self.targets = self.load_images(self.data_path)

    def load_images(self, data_path):
        
        def get_sign_ids(imgname):
            """ Parses an image filename to extract owner, signature ID, and signee information."""

            assert imgname.startswith("NFI-")
            imgname = imgname.lower().split('-')[1].split(".png")[0]
            # print(imgname)
            signee = imgname[:3]
            sign_id = int(imgname[3:-3])
            owner = imgname[-3:]
            return owner, sign_id, signee
        
        def permute_pairs(valid_dict, invalid_dict):
            """ Creates pairs of genuine and forged signature images and assigns labels. """
            
            perm_img_pths = []
            perm_img_labels = []
            for auth, val_imgs in valid_dict.items():
                for val_id, val_pth in val_imgs.items():
                    for inval_id, inval_pth in invalid_dict[auth].items():
                        perm_img_pths.append([val_pth, inval_pth])
                        perm_img_labels.append(1) # label is set to 0 for forged
            
            # permuting for valid image pairs
            for auth, val_imgs in valid_dict.items():
                for val_id, val_pth in val_imgs.items():
                    for val_id2, val_pth2 in valid_dict[auth].items():
                        if val_id!=val_id2:
                            perm_img_pths.append([val_pth, val_pth2])
                            perm_img_labels.append(0) # label is set to 1 for genuine

            perm_img_pths = np.array(perm_img_pths)
            perm_img_labels = np.array(perm_img_labels)
            
            perm_img_pths_flip = perm_img_pths[:, [1, 0]]
            
            perm_img_pths = np.concatenate((perm_img_pths, perm_img_pths_flip), axis=0)
            perm_img_labels = np.concatenate((perm_img_labels, perm_img_labels), axis=0)
            
            print(perm_img_pths[0], perm_img_pths_flip[0])
            return perm_img_pths, perm_img_labels
            
        
        valid_signs_dir = 'genuine_bin'
        invalid_signs_dir = 'forged_bin'
        
        valid_imgs = defaultdict(dict)
        invalid_imgs = defaultdict(dict)
        
        for filename in os.listdir(os.path.join(data_path, valid_signs_dir)):
            owner, sign_id, signee = get_sign_ids(filename)
            valid_imgs[owner][sign_id] = os.path.join(valid_signs_dir, filename)
        for filename in os.listdir(os.path.join(data_path, invalid_signs_dir)):
            owner, sign_id, signee = get_sign_ids(filename)
            invalid_imgs[owner][sign_id] = os.path.join(invalid_signs_dir, filename)
        
        image_pairs, targets = permute_pairs(valid_imgs, invalid_imgs)
        
        return image_pairs, targets
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img1_pth, img2_pth = self.image_pairs[idx]
        img1_pth, img2_pth = os.path.join(self.data_path, img1_pth), os.path.join(self.data_path, img2_pth)
        img1 = transforms.functional.to_tensor(Image.open(img1_pth).convert('L'))
        img2 = transforms.functional.to_tensor(Image.open(img2_pth).convert('L'))
        y = self.targets[idx]
        # y = torch.from_numpy(np.array([target], dtype=np.float32))
        
        return (img1, img2, y)


class SignatureTripletDataset(Dataset):
    """
    Custom dataset for signature verification task.

    Args:
        data_path (str): The root directory containing genuine and forged signature images.
        
    Attributes:
        image_pairs (list): List of image pairs for signature verification.
        targets (list): List of corresponding labels (0 for genuine, 1 for forged).

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Retrieves a pair of images and their label at the specified index.
    """

    def __init__(self, data_path="None"):
        assert data_path != None
        
        self.data_path = data_path
        self.triplets, self.targets = self.load_images(self.data_path)

    def load_images(self, data_path):
        
        def get_sign_ids(imgname):
            """ Parses an image filename to extract owner, signature ID, and signee information."""

            assert imgname.startswith("NFI-")
            imgname = imgname.lower().split('-')[1].split(".png")[0]
            # print(imgname)
            signee = imgname[:3]
            sign_id = int(imgname[3:-3])
            owner = imgname[-3:]
            return owner, sign_id, signee
        
        def permute_triplets(valid_dict, invalid_dict):
            """ Creates triplets of genuine, anchor and forged signature images and assigns labels. """
            perm_img_pths = []
            perm_img_labels = []
            for auth, val_imgs in valid_dict.items():
                for a_val_id, a_val_pth in val_imgs.items(): #anchors
                    for p_val_id, p_val_pth in val_imgs.items(): # positives
                        if a_val_id!=p_val_id:
                            for n_val_id, n_val_pth in invalid_dict[auth].items(): # negatives
                                perm_img_pths.append([p_val_pth, a_val_pth, n_val_pth])
                                perm_img_labels.append([1, 0]) # label is set to 0 for forged
                                
                                # perm_img_pths.append([n_val_pth, a_val_pth, p_val_pth])
                                # perm_img_labels.append([0, 1]) # label is set to 0 for forged
            return perm_img_pths, perm_img_labels

        valid_signs_dir = 'genuine_bin'
        invalid_signs_dir = 'forged_bin'
        
        valid_imgs = defaultdict(dict)
        invalid_imgs = defaultdict(dict)
        
        for filename in os.listdir(os.path.join(data_path, valid_signs_dir)):
            owner, sign_id, signee = get_sign_ids(filename)
            valid_imgs[owner][sign_id] = os.path.join(valid_signs_dir, filename)
        for filename in os.listdir(os.path.join(data_path, invalid_signs_dir)):
            owner, sign_id, signee = get_sign_ids(filename)
            invalid_imgs[owner][sign_id] = os.path.join(invalid_signs_dir, filename)
        
        triplets, targets = permute_triplets(valid_imgs, invalid_imgs)
        return triplets, targets
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img1_pth, img2_pth, img3_pth = self.triplets[idx]
        img1_pth, img2_pth, img3_pth = os.path.join(self.data_path, img1_pth), os.path.join(self.data_path, img2_pth), os.path.join(self.data_path, img3_pth)
        img1 = transforms.functional.to_tensor(Image.open(img1_pth).convert('L'))
        img2 = transforms.functional.to_tensor(Image.open(img2_pth).convert('L'))
        img3 = transforms.functional.to_tensor(Image.open(img3_pth).convert('L'))
        
        y = self.targets[idx]
        # y = torch.from_numpy(np.array([target], dtype=np.float32))
        
        return (img1, img2, img3, y)
        

if __name__ == '__main__':
    data_path = 'dataset/'
    signatureDataset = SignatureContrastiveDataset(data_path=data_path)
    
    print(signatureDataset.image_pairs)