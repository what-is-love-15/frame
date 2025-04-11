import cv2
import numpy as np
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
        return f, None, ''

        # Выбираем самый контрастный контур
    candidates.sort(key=cv2.contourArea, reverse=True)
    best_contour = candidates[0]

    rect = cv2.minAreaRect(best_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    center, size, ang = rect
    center = tuple(map(int, center))
    size = tuple(map(int, size))

    if ang < -45:
        ang += 90

    # контур и повёрнутый прямоугольник
    cv2.drawContours(f, [box], 0, (0, 0, 255), 2)

    # голубой не повернутый прямоугольник
    x, y, w, h = cv2.boundingRect(best_contour)
    padding = 20
    x_pad = max(0, x - padding)
    y_pad = max(0, y - padding)
    w_pad = min(f.shape[1] - x_pad, w + 2 * padding)
    h_pad = min(f.shape[0] - y_pad, h + 2 * padding)
    cv2.rectangle(f, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (255, 255, 0), 2)

    # вывод угла
    text_x, text_y = x + w // 2, y - 30
    cv2.putText(f, f"Angle: {round(ang, 3)}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # вывод текста
    rotation_matrix = cv2.getRotationMatrix2D(center, ang, 1.0)
    rotated = cv2.warpAffine(f, rotation_matrix, (f.shape[1], f.shape[0]))
    x, y = int(center[0] - size[0] / 2), int(center[1] - size[1] / 2)
    w, h = int(size[0]), int(size[1])
    roi_rotated = rotated[y:y + h, x:x + w]
    roi_gray = cv2.cvtColor(roi_rotated, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(roi_thresh) > 127:
        roi_thresh = cv2.bitwise_not(roi_thresh)

    roi_thresh = cv2.resize(roi_thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # проверка ориентации — если текст "вертикальный", разворачиваем
    if roi_thresh.shape[0] > roi_thresh.shape[1] * 1.2:
        roi_thresh = cv2.rotate(roi_thresh, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("ROI for OCR", roi_thresh)

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(roi_thresh, config=custom_config).strip()

    if text:
        print(f'Распознанный текст: {text}')
        cv2.putText(f, text, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    return f, ang, text


video_path = 'video_1.mp4'
use_webcam = False  # или True для доп задания
print('Будем работать с веб-камерой?')
answer = input().lower()
if answer == 'yes':
    use_webcam = True
cap = cv2.VideoCapture(0) if use_webcam else cv2.VideoCapture(video_path)

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

    cv2.imshow('Video stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()
print(f'Самый четкий кадр № {sharpest_frame_number}, резкость: {sharpest_value}')

if sharpest_frame is not None:
    pro_frame, angle, text = detect_gray_square(sharpest_frame)

    if angle is not None:
        print(f'Угол: {angle} градусов')

    cv2.imshow('Detected Gray Square', pro_frame)
    filename = f'frame_{round(sharpest_value)}.jpeg'
    cv2.imwrite(os.path.join(output_dir, filename), pro_frame)
    print(f'Сохранено: {filename}')
    cv2.waitKey(0)
    cv2.destroyAllWindows()