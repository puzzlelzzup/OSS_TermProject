import cv2
import numpy as np

def find_crosswalk(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return

    # HSV로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    # 결과 이미지 출력
    cv2.imshow('Detected Crosswalk', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 이미지 경로 설정 후 함수 호출
find_crosswalk('walk_people.jpg')
