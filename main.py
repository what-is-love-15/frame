import cv2
import numpy as np
import os

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)


def detect_gray_square(f):
    # Переводим в HSV для поиска серого цвета
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)

    # Диапазон серого цвета (настраиваемый)
    lower_gray = np.array([80, 20, 100])
    upper_gray = np.array([120, 150, 220])
    # Вычисляем средний цвет в центре кадра
    h, w = hsv.shape[:2]
    center_x, center_y = w // 2, h // 2
    square_region = hsv[center_y - 10:center_y + 10, center_x - 10:center_x + 10]
    avg_hsv = np.mean(square_region, axis=(0, 1))
    print(f'Средний цвет квадрата (HSV): {avg_hsv}')

    mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # Поиск контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = f.shape[:2]
    min_area = 500
    max_area = 0.3 * w * h  # не более 30% от кадра

    candidates = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            ratio = w / float(h)
            if 0.8 < ratio < 1.2:  # почти квадрат
                candidates.append(c)

    print(f'Найдено контуров: {len(contours)}')
    if not candidates:
        return f, None

        # Выбираем самый контрастный контур
    candidates.sort(key=cv2.contourArea, reverse=True)
    best_contour = candidates[0]

    rect = cv2.minAreaRect(best_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    ang = rect[2]
    if ang < -45:
        ang += 90

    # Рисуем контур и повёрнутый прямоугольник
    # cv2.drawContours(f, [best_contour], -1, (0, 255, 0), 2) зеленый лишний
    cv2.drawContours(f, [box], 0, (0, 0, 255), 2)

    # Вывод угла
    x, y, w, h = cv2.boundingRect(best_contour)
    text_x, text_y = x + w // 2, y - 10

    cv2.putText(f, f"Angle: {round(ang, 3)}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return f, ang


video_path = 'video_3.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print('Ошибка: видео не открывается')
    exit()

sharpest_frame = None
sharpest_value = 0
sharpest_frame_number = 0
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if laplacian_var > sharpest_value:
        sharpest_value = laplacian_var
        sharpest_frame = frame.copy()
        sharpest_frame_number = frame_number

    frame_number += 1

cap.release()
print(f'Самый четкий кадр № {sharpest_frame_number}, резкость: {sharpest_value}')

if sharpest_frame is not None:
    pro_frame, angle = detect_gray_square(sharpest_frame)

    if angle is not None:
        print(f'Angle: {angle} градусов')

    cv2.imshow('Detected Gray Square', pro_frame)
    if angle is not None:
        filename = f'frame_{sharpest_frame_number}_angle_{round(angle, 2)}.png'
    else:
        filename = f'frame_{sharpest_frame_number}_no_angle.png'
    cv2.imwrite(os.path.join(output_dir, filename), pro_frame)
    print(f'Сохранено: {filename}')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
