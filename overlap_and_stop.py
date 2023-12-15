import cv2
import numpy as np

img = cv2.imread('walk_people.jpg')
height, width, channel = img.shape
stop=0

print('original image shape:', height, width, channel)
if img is None:
    print(f"Error: Unable to read the image")

# 횡단보도 인식
# HSV로 변환
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 빨간색 범위 설정
lower_red = np.array([0, 0, 200])
upper_red = np.array([255, 255, 255])
# 빨간색 마스크 생성
mask = cv2.inRange(hsv, lower_red, upper_red)
# 블러 적용
blurred = cv2.GaussianBlur(mask, (5, 5), 0)
# 컨투어 찾기
contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
crosswalk_rectangles = []
# 횡단보도 영역 찾기
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4 and cv2.contourArea(contour) > 500:
        # 횡단보도인 경우
        x, y, w, h = cv2.boundingRect(approx)
        crosswalk_rectangles.append((x, y, x+w, y+h))
if crosswalk_rectangles:
    # 모든 횡단보도 블록의 경계상자를 통해 전체 횡단보도의 경계상자 계산
    min_x = min(rect[0] for rect in crosswalk_rectangles)
    min_y = min(rect[1] for rect in crosswalk_rectangles)
    max_x = max(rect[2] for rect in crosswalk_rectangles)
    max_y = max(rect[3] for rect in crosswalk_rectangles)
    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

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

# 횡단보도와 사람이 겹치는지 확인 후 초록색 사각형으로 표시
for i in range(len(merged_person_boxes)):
    x1, y1, w, h = merged_person_boxes[i] 
    x2, y2 = x1+w, y1+h
    x3, y3 = min_x, min_y 
    x4, y4 = max_x, max_y
    ## 오른쪽으로 벗어나 있는 경우
    if x2 < x3:
        print(f'case1 {x2} {x3}')
        continue
    ## 왼쪽으로 벗어나 있는 경우
    if x1 > x4:
        print(f'case2 {x1} {x4}')
        continue
    ## 위쪽으로 벗어나 있는 경우
    if  y2 < y3:
        print(f'case3 {y2} {y3}')
        continue
    ## 아래쪽으로 벗어나 있는 경우
    if  y1 > y4:
        print(f'case4 {y1} {y4}')
        continue
    stop+=1
    left_up_x = max(x1, x3)
    left_up_y = max(y1, y3)
    right_down_x = min(x2, x4)
    right_down_y = min(y2, y4)
    cv2.rectangle(img, (left_up_x, left_up_y), (right_down_x, right_down_y), (0,255,0), 2)
    print(f' {left_up_x}, {left_up_y}, {right_down_x}, {right_down_y}')
    
# 인식한 차들에 사각형 그린 후 사람이 횡단보도에 있다면 stop 출력
for i in range(len(merged_car_boxes)):
    x, y, w, h = merged_car_boxes[i]
    print(f'class car detected at {x}, {y}, {w}, {h}')
    color = colors[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, 'car', (x, y - 10), font, 1, color, 2)
    if(stop>0):
        cv2.putText(img, 'stop', (int(x+w/2)-30, int(y+h/2)), font, 1, (0,0,255), 2)

cv2.imshow('Objects', img)
cv2.waitKey()
cv2.destroyAllWindows()