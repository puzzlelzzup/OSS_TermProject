import cv2
import numpy as np

# read image
img = cv2.imread('./image/walk_people.jpg')
height, width, channel = img.shape
print('original image shape:', height, width, channel)

# get blob from image
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
print('blob shape:', blob.shape)

# read coco object names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print('number of classes =', len(classes))

# load pre-trained yolo model from configuration and weight files
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# set output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print('output layers:', output_layers)

# detect objects
net.setInput(blob)
outs = net.forward(output_layers)

# get bounding boxes and confidence scores
class_ids = []
confidence_scores = []
boxes = []

for out in outs: # for each detected object
    for detection in out: # for each bounding box
        scores = detection[5:] # scores (confidence) for all classes
        class_id = np.argmax(scores) # class id with the maximum score (confidence)
        confidence = scores[class_id] # the maximum score

        if confidence > 0.5 and classes[class_id] != 'backpack': # except backpack detect
            # bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidence_scores.append(float(confidence))
            class_ids.append(class_id)

print('number of dectected objects =', len(boxes))

# non maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, 0.5, 0.4)
print('number of final objects =', len(indices))

# create list
person_boxes = []
car_boxes = []

if len(indices) > 0:
    indices = indices.flatten()
    for i in indices:
        x, y, w, h = boxes[i]
        class_id = class_ids[i]

        if classes[class_id] == 'person':
            person_boxes.append([x, y, w, h])
        elif classes[class_id] == 'car':
            car_boxes.append([x, y, w, h])

def merge_boxes(boxes):
    merged_boxes = []
    for box in boxes:
        x, y, w, h = box
        merged = False

        for mb in merged_boxes:
            mb_x, mb_y, mb_w, mb_h = mb

            # check if it overlaps
            if x < mb_x + mb_w and x + w > mb_x and y < mb_y + mb_h and y + h > mb_y:
                # If it overlaps, merge the boxes
                merged = True
                mb[0] = min(x, mb_x)
                mb[1] = min(y, mb_y)
                mb[2] = max(x + w, mb_x + mb_w) - mb[0]
                mb[3] = max(y + h, mb_y + mb_h) - mb[1]
                break

        if not merged:
            merged_boxes.append([x, y, w, h])

    return merged_boxes

merged_person_boxes = merge_boxes(person_boxes)
merged_car_boxes = merge_boxes(car_boxes)

                
# draw bounding boxes with labels on image
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_TRIPLEX

# printf
for i in range(len(merged_person_boxes)):
    x, y, w, h = merged_person_boxes[i]
    print(f'class person detected at {x}, {y}, {w}, {h}')
    color = colors[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, 'person', (x, y - 10), font, 1, color, 2)
    
for i in range(len(merged_car_boxes)):
    x, y, w, h = merged_car_boxes[i]
    print(f'class car detected at {x}, {y}, {w}, {h}')
    color = colors[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, 'car', (x, y - 10), font, 1, color, 2)

cv2.imshow('Objects', img)
cv2.waitKey()
cv2.destroyAllWindows()
