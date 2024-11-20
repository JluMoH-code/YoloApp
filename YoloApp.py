import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
import numpy as np

class YOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = None
        self.current_index = 0

    def initUI(self):
        self.setWindowTitle('YOLO Detector App')

        # Кнопки для загрузки весов, выбора файлов и навигации
        self.load_weights_button = QPushButton('Load Weights', self)
        self.load_weights_button.clicked.connect(self.load_weights)

        self.load_file_button = QPushButton('Load Image/Video/Folder', self)
        self.load_file_button.clicked.connect(self.load_file)
        
        self.stop_button = QPushButton("Stop Video", self)
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)  # Неактивна, пока видео не запущено

        # Области для отображения изображений
        self.original_label = QLabel('Original Image/Video', self)
        self.processed_label = QLabel('Processed Image/Video', self)

        hbox = QHBoxLayout()
        hbox.addWidget(self.original_label)
        hbox.addWidget(self.processed_label)

        layout = QVBoxLayout()
        layout.addWidget(self.load_weights_button)
        layout.addWidget(self.load_file_button)
        layout.addWidget(self.stop_button) 
        layout.addLayout(hbox)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_weights(self):
        weights_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Weights", "", "Weights Files (*.pt)")
        if weights_path:
            self.model = YOLO(weights_path)
            print("Weights loaded from:", weights_path)

            # Показ уведомления об успешной загрузке весов
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText("Weights loaded successfully!")
            msg_box.setInformativeText(f"Weights loaded from: {weights_path}")
            msg_box.setWindowTitle("Load Weights")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()

    def load_file(self):
        # Загрузка изображения, видео
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image, Video, or Folder", "", "Image/Video Files (*.jpg *.jpeg *.png *.bmp *.mp4)")
        if file_path:
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.process_image(file_path)
            elif file_path.endswith('.mp4'):
                self.start_video_processing(file_path)

    def process_image(self, file_path):
        img = cv2.imread(file_path)

        # Обработка с использованием YOLO
        if self.model:
            results = self.model(img)  # Обработка уменьшенного изображения
            processed_img = self.draw_boxes(img.copy(), results)  
            resized_img = self.resize_image_to_fit(processed_img)
            self.display_image(resized_img, self.processed_label)

        # Загрузка и отображение исходного изображения
        resized_img = self.resize_image_to_fit(img)
        self.display_image(resized_img, self.original_label)        

    def resize_image_to_fit(self, img, max_width=640, max_height=480):
        # Масштабирование изображения, чтобы оно помещалось
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    def draw_boxes(self, img, results):
        # Функция для рисования рамок на изображении
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты рамок
                conf = box.conf[0]  # Уверенность предсказания
                cls = int(box.cls[0])  # Класс обнаруженного объекта

                # Рисуем рамки и метки
                label = f"{self.model.names[cls]} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img


    def start_video_processing(self, file_path):
        # Запуск видео в отдельном потоке
        self.video_thread = VideoThread(file_path, self.model)
        self.video_thread.original_frame.connect(lambda frame: self.display_image(frame, self.original_label))
        self.video_thread.processed_frame.connect(lambda frame: self.display_image(frame, self.processed_label))
        self.video_thread.finished.connect(self.on_video_finished)  # Сигнал завершения потока
        self.stop_button.setEnabled(True)  # Активируем кнопку остановки
        self.video_thread.start()

    def stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        self.stop_button.setEnabled(False)  # Деактивируем кнопку остановки

    def on_video_finished(self):
        # Сброс отображения по завершению видео
        self.original_label.clear()
        self.processed_label.clear()
        self.stop_button.setEnabled(False)

    def display_image(self, img, label):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_image))

class VideoThread(QThread):
    original_frame = pyqtSignal(np.ndarray)
    processed_frame = pyqtSignal(np.ndarray)

    def __init__(self, file_path, model):
        super().__init__()
        self.file_path = file_path
        self.model = model
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.file_path)
        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Обработка кадра с использованием YOLO
            if self.model:
                results = self.model(frame)
                processed_frame = self.draw_boxes(frame.copy(), results)
                resized_frame = self.resize_frame_to_fit(processed_frame)
                self.processed_frame.emit(resized_frame)

            # Уменьшаем размер кадра для показа
            resized_frame = self.resize_frame_to_fit(frame)
            self.original_frame.emit(resized_frame)

            QThread.msleep(30)  # задержка для обновления кадров

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def resize_frame_to_fit(self, frame, max_width=640, max_height=480):
        # Масштабирование кадра для показа в окне
        h, w = frame.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    def draw_boxes(self, img, results):
        # Функция для рисования рамок на кадре
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                # Рисуем рамки и метки
                label = f"{self.model.names[cls]} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = YOLOApp()
    mainWin.show()
    sys.exit(app.exec_())
