# 驾驶员疲劳检测系统
# @author : 杨雪清
# @date: 2024年5月

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import threading
import sys
import cv2
import dlib
import pickle

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# 窗口主类

class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化设置
        # 继承父类方法进行初始化
        super().__init__()
        self.setWindowTitle('驾驶员疲劳检测系统')
        self.resize(750, 750)
        self.setWindowIcon(QIcon("UI_images/logo.png"))
        self.video_capture = cv2.VideoCapture(0)  # 打开电脑默认摄像头
        # 初始化中止事件
        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        # 要检测的视频名称
        self.source = ''
        # 加载lbp检测器
        # 加载人脸识别模型
        self.initUI()
        self.set_down()

    # 初始化界面
    def initUI(self):

        # 设置字体
        font_v = QFont('Microsoft YaHei', 14)
        generally_font = QFont('Microsoft YaHei', 16)

        # ---------视频识别界面----------
        video_widget = QWidget()
        video_layout = QVBoxLayout()

        # 设置视频识别区的标题
        self.video_title = QLabel("驾驶员疲劳检测系统")
        self.video_title.setFont(font_v)
        self.video_title.setAlignment(Qt.AlignCenter)
        self.video_title.setFont(generally_font)

        # 设置显示的界面
        self.DisplayLabel = QLabel()
        self.DisplayLabel.setPixmap(QPixmap(""))
        self.btn_open_rsmtp = QPushButton("调用镜头")
        self.btn_open_rsmtp.setFont(font_v)
        # 设置打开摄像头的按钮和样式
        self.btn_open_rsmtp.setStyleSheet("QPushButton{color:white}"
                                          "QPushButton:hover{background-color: rgb(61,145,64);}"
                                          "QPushButton{background-color:rgb(0,201,87)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:5px}"
                                          "QPushButton{padding:5px 5px}"
                                          "QPushButton{margin:5px 5px}")
        # 设置结束演示的按钮和样式
        self.btn_close = QPushButton("停止识别")
        self.btn_close.setFont(font_v)
        self.btn_close.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(255,0,0);}"
                                     "QPushButton{background-color:rgb(255,99,71)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        # 将组件添加到布局上
        self.btn_open_rsmtp.clicked.connect(self.open_local)
        # self.btn_open_rsmtp.clicked.connect(self.display_video())
        self.btn_close.clicked.connect(self.close)

        video_layout.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_title)
        video_layout.addWidget(self.DisplayLabel)
        self.DisplayLabel.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.btn_open_rsmtp)
        video_layout.addWidget(self.btn_close)
        video_widget.setLayout(video_layout)

        # ---------关于界面----------
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('\n\n\n\n\n\n欢迎使用驾驶员疲劳检测系统 1.0\n\n开发人员：上上签')
        about_title.setFont(QFont('Microsoft YaHei', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setAlignment(Qt.AlignCenter)

        label_super = QLabel()
        label_super.setFont(QFont('Microsoft YaHei', 16))
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()  # 平分空白
        about_layout.addWidget(about_img)
        about_layout.addStretch()  # 平分空白
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)
        # 添加2个子页面
        self.addTab(video_widget, '疲劳检测')
        self.addTab(about_widget, '关于系统')
        self.setTabIcon(1, QIcon('UI_images/视频.png'))
        self.setTabIcon(2, QIcon('UI_images/logo_about.png'))

    # 关闭程序 询问用户是否退出
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)  # “退出”显示在标题栏，“是否要退出程序“显示在消息框，代表默认焦点在NO上
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    def open_local(self):
        # 选择录像文件进行读取
        mp4_filename = 0
        self.source = mp4_filename
        # 读取摄像头进行实时得显示
        self.video_capture = cv2.VideoCapture(self.source)
        th = threading.Thread(target=self.display_video)  # 人脸识别主进程
        th.start()

    # 退出进程
    def close(self):
        # 点击关闭按钮后重新初始化界面
        self.stopEvent.set()
        self.set_down()

    # 人脸识别的主进程
    def display_video(self):
        # 首先把打开按钮关闭
        self.btn_close.setEnabled(True)
        process_this_frame = True

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
        while True:
            ret, frame = self.video_capture.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false,frame为视频的每一帧图像
            frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。

            if process_this_frame:
                # 定义SVM模型
                with open('svm_model.pkl', 'rb') as f:
                    svm = pickle.load(f)

                # 定义人脸识别器
                haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

                # 定义dlib人脸关键点检测器
                predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

                # 灰度化
                # opencv的图像是BGR格式的，而我们需要是的RGB格式的，因此需要进行一个转换。
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 人脸检测
                faces = haar_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    # 转换为dlib.rectangle
                    rect = dlib.rectangle(left=int(x), top=int(y), right=int(x + w), bottom=int(y + h))

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
                        print("疲劳驾驶!!!!!!!!!!")
                        # 保存图片并进行实时的显示
                        frame = frame
                        frame_height = frame.shape[0]
                        frame_width = frame.shape[1]
                        frame_scale = 500 / frame_height
                        frame_resize = cv2.resize(frame,
                                                  (int(frame_width * frame_scale), int(frame_height * frame_scale)))
                        # 将捕捉到的人脸显示出来
                        cv2.putText(frame_resize, "Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imwrite("Pictures/Alert.png", frame_resize)
                        self.DisplayLabel.setPixmap(QPixmap("Pictures/Alert.png"))  # 显示图片

                    else:
                        print("非疲劳驾驶")
                        # 保存图片并进行实时的显示
                        frame = frame
                        frame_height = frame.shape[0]
                        frame_width = frame.shape[1]
                        frame_scale = 500 / frame_height
                        frame_resize = cv2.resize(frame,
                                                  (int(frame_width * frame_scale), int(frame_height * frame_scale)))
                        # 将捕捉到的人脸显示出来
                        cv2.putText(frame_resize, "Normal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imwrite("Pictures/Normal.png", frame_resize)
                        self.DisplayLabel.setPixmap(QPixmap("Pictures/Normal.png"))  # 显示图片
            process_this_frame = not process_this_frame

        # 识别操作完成 关闭【停止识别】按钮
        self.btn_close.setEnabled(True)
        self.set_down()

    # 初始化视频检测界面
    def set_down(self):
        self.video_capture.release()  # 发布软件资源，释放硬件资源
        cv2.destroyAllWindows()  # 关闭窗口并取消分配任何相关的内存使用
        self.DisplayLabel.setPixmap(QPixmap("UI_images/black.jpeg"))


if __name__ == "__main__":
    # 加载页面
    app = QApplication(sys.argv)  # 参数默认为空列表
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())  # 消息循环结束之后返回0，接着调用sys.exit(0)退出程序






