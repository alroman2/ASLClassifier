import cv2
import glob
import os
import numpy as np
import torchvision.ops as bops
import torch
import re
from metrics import Metric
from model import create_model
from xml.etree import ElementTree as ET
import gc

torch.cuda.empty_cache()
gc.collect()

DIR_IMAGES = '/backend/test_data'
DIR_IMAGES_OUT = '/backend/Inference_out'

test_images = glob.glob(f"{DIR_IMAGES}/*.jpg")

CLASSES = ['background', 'I love you', 'Thank You', 'Yes']

DETECTION_THRESHOLD = 0.5

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

model = create_model(num_classes=4).to(device)
model.load_state_dict(torch.load(
    '/backend/Final Graph Data/100_Epochs/ASL-DETECTION-RESNET50_94.pth'))
model.eval()


def ground_truth(image_path):
    # pattern = re.compile(
    #    r"[a-zA-Z]+\.[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\.", re.IGNORECASE)
    #image_name = pattern.match(image_path.split('/')[-1]).group(0)
    #image_name = image_path.split('/')[-1].split('.')[0]
    # xml_path = f'{DIR_IMAGES}/{image_name}xml'
    # get the bounding boxes from the xml file
    image_path = image_path.split('/')[-1]
    file_name = image_path[:-4]+'.xml'
    xml_path = os.path.join(DIR_IMAGES, file_name)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
    # if no boxes are found, return an 0 value, the worst iou

    return boxes


def calculate_iou(ground_truth_box_array, predicted_box_array):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    if predicted_box_array is None or len(predicted_box_array) == 0:
        return 0.0

    ground_truth_box_array = ground_truth_box_array[0]
    bb1 = {
        'x1': ground_truth_box_array[0],
        'y1': ground_truth_box_array[1],
        'x2': ground_truth_box_array[2],
        'y2': ground_truth_box_array[3]
    }
    bb2 = {
        'x1': predicted_box_array[0],
        'y1': predicted_box_array[1],
        'x2': predicted_box_array[2],
        'y2': predicted_box_array[3]
    }

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box.
    # NOTE: We MUST ALWAYS add +1 to calculate area when working in
    # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
    # is the bottom right pixel. If we DON'T add +1, the result is wrong.
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


if __name__ == '__main__':
    model_metric = Metric()

    for i in range(len(test_images)):
        image_name = test_images[i].split('/')[-1].split('.')[0]
        image = cv2.imread(test_images[i])
        original_image = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # get the ground truth boxes
        ground_truth_boxes = ground_truth(test_images[i])

        # make pixel range 0-1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float32).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()

        # filter out boxes with score less than threshold
        boxes = boxes[scores >= DETECTION_THRESHOLD].astype(np.int32)
        draw_boxes = boxes.copy()
        # get predicted classes
        pred_classes = [CLASSES[i]
                        for i in outputs[0]['labels'].cpu().numpy()]
        # draw boxes
        if len(draw_boxes) != 0:
            for j, box in enumerate(draw_boxes):
                cv2.rectangle(original_image, (box[0], box[1]),
                              (box[2], box[3]), (0, 0, 255), 2)
                cv2.putText(original_image, pred_classes[j], (box[0], box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, lineType=cv2.LINE_AA)
                print(f"Image {image_name} : {pred_classes[j]}")
                # calculate iou for each box
            iou = calculate_iou(ground_truth_boxes, box)
            model_metric.update(ground_truth_boxes, box)
            print(f"Image {image_name} iou to ground truth: {iou}")
            print(f"Precision: {model_metric.precision()}")
            print(f"Recall: {model_metric.recall()}")
        else:
            iou = calculate_iou(ground_truth_boxes, None)
            model_metric.update(ground_truth_boxes, None)
            print(f"Image {image_name} iou to ground truth: {iou}")
            print(f"Precision: {model_metric.precision()}")
            print(f"Recall: {model_metric.recall()}")
        cv2.imwrite(f'{DIR_IMAGES_OUT}/{image_name}.jpg', original_image)
        print(f"Image {i+1} done")
