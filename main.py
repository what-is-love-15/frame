import cv2
import numpy as np

video_path = 'video_1.mp4'
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
        break  # если закончилось видео

    # перевод кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # расчет резкости
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if laplacian_var > sharpest_value:
        sharpest_value = laplacian_var
        sharpest_frame = frame.copy()
        sharpest_frame_number = frame_number

    frame_number += 1

cap.release()
print(f'Самый четкий кадр № {sharpest_frame_number}, резкость: {sharpest_value}')

if sharpest_frame is not None:
    cv2.imshow('Sharpest frame', sharpest_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
