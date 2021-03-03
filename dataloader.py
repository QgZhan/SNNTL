
import os
import random
import threading
import numpy as np

import torch
from PIL import Image

from torchvision import datasets, transforms
import torch.utils.data as data
from torchvision.datasets.folder import *
from prefetch_generator import BackgroundGenerator
from args import get_parser

parser = get_parser()
args = parser.parse_args()

img_dict = {'0': 'dog', '1': 'elephant', '2': 'giraffe', '3': 'guitar', '4': 'horse', '5': 'house', '6': 'person'}


class MyThread (threading.Thread):
    def __init__(self, func, thread_idx, idx_list, start, get_type):
        threading.Thread.__init__(self)
        self.func = func
        self.threadID = thread_idx
        self.idx_list = idx_list
        self.start_idx = start
        self.get_type = get_type

    def run(self):
        for idx in self.idx_list:
            self.func(idx, self.start_idx, self.get_type)
            self.start_idx += 1


class ClassMatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset_name, num_classes=10):
        self.target_transform = transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.RandomCrop(227),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0, 0, 0],
                                  std=[1, 1, 1])
             ]
        )
        self.root = root
        self.dataset_name = dataset_name

        self.classes_len = []
        self.data_name_idx = []
        self.data_list = []

        self.image_idx = None
        self.image_list = None
        self.image_label_list = None

        self.path = os.path.join(self.root, self.dataset_name)

        self.iter_idx = [0 for _ in range(num_classes)]
        for i in range(len(img_dict)):
            folder_name = img_dict[str(i)]
            file_data_names = os.listdir(os.path.join(self.path, folder_name))

            random.shuffle(file_data_names)
            self.classes_len.append(len(file_data_names))
            self.data_name_idx.append(file_data_names)
            image_list, _ = self.multi_process_read([int(i)] * len(file_data_names),
                                                    num_thread=10, get_type="path")
            self.data_list.append(image_list)

        self.iter_idx = [0 for _ in range(num_classes)]

    def multi_process_read(self, folder_index_list, num_thread=5, get_type="tensor"):
        self.image_idx = []
        self.image_list = []
        self.image_label_list = []

        sub_len = int(len(folder_index_list) / num_thread)

        th_list = []
        # 创建新线程和添加线程到列表
        for i in range(num_thread):
            start = i * sub_len
            end = (i + 1) * sub_len
            if i == num_thread - 1:
                end = len(folder_index_list)
            sub_list = folder_index_list[start: end]
            thread = MyThread(self.get_item, 0, sub_list, start, get_type)
            th_list.append(thread)  # 添加线程到列表

        # 循环开启线程
        for th in th_list:
            th.start()

        # 等待所有线程完成
        for th in th_list:
            th.join()

        self.image_list = [x for _, x in sorted(zip(self.image_idx, self.image_list))]
        self.image_label_list = [x for _, x in sorted(zip(self.image_idx, self.image_label_list))]
        return self.image_list, self.image_label_list

    def get_item(self, folder_index, idx, get_type="tensor"):
        folder_name = img_dict[str(folder_index)]

        if get_type == "tensor":
            image = self.data_list[folder_index][self.iter_idx[folder_index]]

        elif get_type == "path":
            index = self.data_name_idx[folder_index][self.iter_idx[folder_index]]
            data_path = os.path.join(self.path, folder_name, index)
            image = self.pil_loader(data_path)
            image = self.target_transform(image)

        self.iter_idx[folder_index] += 1
        if self.iter_idx[folder_index] == self.classes_len[folder_index]:
            self.iter_idx[folder_index] = 0

        self.image_idx.append(idx)
        self.image_list.append(image)
        self.image_label_list.append(folder_index)
        return image

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return sum(self.classes_len)


class DataloaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def load_all_data(root_path, directory, batch_size):
    """
    导入所有数据
    source_train_loader = load_all_data("./data/PACS/", "cartoon", batch_size)
    :param root_path: 总数据集所在路径，如: ./data/PACS/
    :param directory: 需提取的数据集名，如：cartoon
    :param batch_size: 批量大小
    :return: 返回torch的DataLoader类
    """
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),  # big: 256, small: 32
         transforms.RandomCrop(227),  # big: 227, small 32
         transforms.ToTensor(),
         transforms.Normalize(mean=[0, 0, 0],
                              std=[1, 1, 1])
         ]
    )
    data = datasets.ImageFolder(root=os.path.join(root_path, directory),
                                transform=transform)
    loader = DataloaderX(data, batch_size=batch_size, shuffle=True,
                         drop_last=False, num_workers=2, pin_memory=False)
    return loader

