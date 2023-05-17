
import torchvision.transforms as transforms


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from PIL import Image

class InvoiceDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        # 存储图片所在的文件夹路径
        self.img_dir = img_dir
        # 存储标签所在的文件夹路径
        self.label_dir = label_dir
        # 使用 os.listdir 函数获取图片文件夹中的所有文件名，并按字母序排序
        self.img_names = sorted(os.listdir(img_dir))
        # 创建一个字典，用于将标签文件名映射到标签 ID
        self.label_map = {f"{i+1}.png": i for i in range(14)}
        # 定义图片变换操作
        self.transform = transforms.Compose([
            transforms.Resize((32,32)),  # 将图片缩放到指定大小
            transforms.ToTensor(),  # 将 PIL 图像转换为 PyTorch 张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        ])

    def __len__(self):
        # 返回数据集中所有样本的数量
        return len(self.img_names)

    def __getitem__(self, idx):
        # 获取指定索引的样本
        img_name = self.img_names[idx]
        # 从图片文件名中获取标签文件名
        label_name = f"{img_name.split('-')[0]}.png"
        # 将标签文件名映射到标签 ID
        label_id = self.label_map[label_name]

        # 构造图片文件的完整路径
        img_path = os.path.join(self.img_dir, img_name)
        # 打开图片文件，并将其转换为 RGB 模式的 PIL 图像
        img = Image.open(img_path).convert("RGB")
        # 对图像进行预处理，并将其转换为 PyTorch 张量
        img_tensor = self.transform(img)

        # 返回图像张量和标签 ID
        return torch.tensor(img_tensor), torch.tensor(label_id)

# 定义图片文件夹和标签文件夹的路径
img_dir = "img"
label_dir = "label"

img_dir1 = "TestImg2"