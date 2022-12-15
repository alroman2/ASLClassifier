import torch
import cv2
import numpy as np

import os
import glob as glob

from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE, BASE_DIR
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform


class SignLanguageDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        print(
            f"Creating dataset with properties: {dir_path} {width} {height} {classes} {transforms}")
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        # get all image files, sorting them to0
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split(
            '/')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # load images ad masks
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # read image
        image = cv2.imread(image_path)

        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resize = cv2.resize(image, (self.width, self.height))
        image_resize /= 255.0

        # capture the xml file for geeting the annotations
        annot_filename = image_name[:-4]+'.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # get size of image
        image_width = image.shape[1]
        image_height = image.shape[0]

        # box coordiantes for xml files are extracted and corrected for image size
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            xmin = int(member.find('bndbox').find('xmin').text)

            xmax = int(member.find('bndbox').find('xmax').text)

            ymin = int(member.find('bndbox').find('ymin').text)

            ymax = int(member.find('bndbox').find('ymax').text)

            # resize the box coordinates
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        # final target
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # apply transforms
        if self.transforms:
            sample = self.transforms(
                image=image_resize,
                bboxes=target["boxes"],
                labels=labels)
            image_resize = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])

        return image_resize, target

    def __len__(self):
        return len(self.all_images)


# prepare the final dataset and dataloaders
train_dataset = SignLanguageDataset(
    TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
valid_dataset = SignLanguageDataset(
    VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn


)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Valid dataset size: {len(valid_dataset)}")


if __name__ == '__main__':
    dataset = SignLanguageDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES)
    print(f"Dataset size: {len(dataset)}")

    def visualize_sample(image, target):
        box = target["boxes"][0]
        label = CLASSES[target['labels']]

        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 0), 1
        )

        cv2.putText(
            image,
            label,
            (int(box[0]), int(box[1]-5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255), 2
        )
        image *= 255
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{BASE_DIR}/Temp/{1}.jpg', image)

    NUM_SAMPLES_TO_VISUALIZE = 15
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        print(f"Visualizing sample at index: {i}")
        image, target = dataset[i]
        visualize_sample(image, target)
