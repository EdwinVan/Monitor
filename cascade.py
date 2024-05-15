from datetime import time

import cv2
import dlib
import numpy as np
from sklearn import svm
import pickle

# 定义人脸识别器
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 定义dlib人脸关键点检测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# 定义EAR和MAR计算函数
def ear(eye_landmarks):
    denominator = eye_landmarks[3].y - eye_landmarks[2].y
    if denominator == 0:
        return 0
    else:
        return (eye_landmarks[1].y - eye_landmarks[0].y) / denominator


def mar(mouth_landmarks):
    denominator = mouth_landmarks[16].y - mouth_landmarks[8].y
    if denominator == 0:
        return 0
    else:
        return (mouth_landmarks[12].y - mouth_landmarks[0].y) / denominator


# 定义SVM模型
with open('svm_model.pkl', 'rb') as f:
    svm = pickle.load(f)


# 视频采集
cap = cv2.VideoCapture(0)


while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # 转换为dlib.rectangle
        rect = dlib.rectangle(left=int(x), top=int(y), right=int(x+w), bottom=int(y+h))

        # 人脸关键点检测
        shape = predictor(gray, rect)

        # 眼睛和嘴巴的特征点
        eye_landmarks = shape.parts()[36:48]
        mouth_landmarks = shape.parts()[48:68]

        # 计算EAR和MAR值
        ear_value = ear(eye_landmarks)
        mar_value = mar(mouth_landmarks)

        # 预测驾驶员状态
        prediction = svm.predict([[ear_value, mar_value]])

        # 显示结果
        if prediction == 0:
            cv2.putText(frame, "Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite("Pictures/Alert.png", frame)
            print("疲劳驾驶!!!!!!!!!!")
            break

        else:
            cv2.putText(frame, "Normal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite("Pictures/Normal.png", frame)
            print("非疲劳驾驶")
