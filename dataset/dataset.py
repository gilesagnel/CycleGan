import os
import torch.utils.data as data
from PIL import Image
import random
import torchvision.transforms as transforms
import shutil
import requests


class UnPairedDataset(data.Dataset):
    def __init__(self, opt):
        self.root = opt.data_root
        self.phase = "train" if opt.isTrain else "test"

        self.crop_size = opt.crop_size
        self.load_size = opt.load_size
        self.preprocess = opt.preprocess # crop and resize

        self.dir_A = os.path.join(self.root, f"{self.phase}A")  
        self.dir_B = os.path.join(self.root, f"{self.phase}B") 

        self.A_paths = sorted(make_dataset(self.dir_A))   
        self.B_paths = sorted(make_dataset(self.dir_B))    
        self.A_size = len(self.A_paths)  
        self.B_size = len(self.B_paths)    
        self.transform_A = get_transform(self.preprocess, self.load_size, self.crop_size)
        self.transform_B = get_transform(self.preprocess, self.load_size, self.crop_size)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  
        B_path = random.choice(self.B_paths)
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
    

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images
    

def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png']  
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_transform(preprocess, load_size=None, crop_size=None):
    transform_list = []
    if preprocess:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, transforms.InterpolationMode.BICUBIC))
        transform_list.append(transforms.RandomCrop(crop_size))
        
    
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def download_dataset():
    TOP_FOLDER = "pajama-fy"
    SYNSET_IDS = ["n02802426", "n03877472"]
    DIR_NAMES = ["A", "B"]

    os.makedirs(os.path.join("dataset", TOP_FOLDER), exist_ok=True)

    for synset_id, dir_name in zip(SYNSET_IDS, DIR_NAMES):
        url = f"https://image-net.org/data/winter21_whole/{synset_id}.tar"
        tar_file = os.path.join("dataset", f"{synset_id}.tar")
        target_dir = os.path.join("dataset", TOP_FOLDER, f"train{dir_name}")
        os.makedirs(target_dir)
        response = requests.get(url)
        with open(tar_file, "wb") as f:
            f.write(response.content)

        shutil.unpack_archive(tar_file, target_dir)
        os.remove(tar_file)

        test_dir = os.path.join("dataset", TOP_FOLDER, f"test{dir_name}") 
        os.makedirs(test_dir)

        files = os.listdir(target_dir)
        files_to_move = random.sample(files, int(len(files) * 0.2))
        for file_name in files_to_move:
            src = os.path.join(target_dir, file_name)
            dst = os.path.join(test_dir, file_name)
            shutil.move(src, dst) 