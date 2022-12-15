import cv2
import glob
import os
import numpy as np
import torch

from model import create_model

DIR_IMAGES = '/backend/test_data'
DIR_IMAGES_OUT = '/backend/Inference_out'

test_images = glob.glob(f"{DIR_IMAGES}/*")

CLASSES = ['background', 'I love you', 'Thank You', 'Yes']

DETECTION_THRESHOLD = 0.5

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

model = create_model(num_classes=4).to(device)
model.load_state_dict(torch.load(
    '/backend/Final Graph Data/100_Epochs/ASL-DETECTION-RESNET50_94.pth'))
model.eval()

for i in range(len(test_images)):
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    original_image = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

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

    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()

        # filter out boxes with score less than threshold
        boxes = boxes[scores >= DETECTION_THRESHOLD].astype(np.int32)
        draw_boxes = boxes.copy()
        # get predicted classes
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        # draw boxes
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(original_image, (box[0], box[1]),
                          (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(original_image, pred_classes[j], (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, lineType=cv2.LINE_AA)
            print(f"Image {image_name} : {pred_classes[j]}")

        cv2.imwrite(f'{DIR_IMAGES_OUT}/{image_name}.jpg', original_image)
        print(f"Image {i+1} done")
