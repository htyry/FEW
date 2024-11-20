import csv
import os

from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def gray_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("P")


# 一个简单的构造数据集类
class GeneralDataset(Dataset):
    """
    A general dataset class.
    """

    def __init__(
        self,
        root="",
        mode="train",
        loader=pil_loader,
        transforme=None
    ):
        super(GeneralDataset, self).__init__()
        assert mode in [
            "train",
            "val",
            "test",
        ], "mode must be in ['train', 'val', 'test']"

        self.root = root
        self.mode = mode
        self.loader = loader
        self.trfms = transforme

        self.data_list, self.label_list, self.class_label_dict = self.generate_data_list()

        self.label_num = len(self.class_label_dict)
        self.length = len(self.data_list)

        print(
            "load {} {} image with {} label.".format(self.length, mode, self.label_num)
        )

    def generate_data_list(self):

        meta_csv = os.path.join(self.root, "{}.csv".format(self.mode))

        data_list = []
        label_list = []
        class_label_dict = dict()
        with open(meta_csv) as f_csv:
            f_train = csv.reader(f_csv, delimiter=",")
            for row in f_train:
                if f_train.line_num == 1:
                    continue
                image_name, image_class = row
                if image_class not in class_label_dict:
                    class_label_dict[image_class] = len(class_label_dict)
                image_label = class_label_dict[image_class]
                data_list.append(image_name)
                label_list.append(image_label)

        return data_list, label_list, class_label_dict


    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        image_name = self.data_list[idx]
        image_path = os.path.join(self.root, "images", image_name)
        data = self.loader(image_path)

        if self.trfms is not None:
            data = self.trfms(data)
        label = self.label_list[idx]

        return data, label