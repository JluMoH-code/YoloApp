import cv2
from PyQt5.QtGui import QImage

class Utilities:
    @staticmethod
    def draw_results(image, results, names):
        image_with_results = image.copy()
        for result in results:
            for box in result.boxes:
                image_with_results = Utilities.draw_boxes_with_labels(image, box, names)
                
        return image_with_results

    @staticmethod
    def draw_boxes_with_labels(image, boxes, names):
        image = Utilities.draw_box(image, boxes)
        label = Utilities.get_label(boxes, names)
        image = Utilities.draw_label(image, boxes, label)
        return image

    @staticmethod
    def draw_box(image, bbox, color=(0, 255, 0), thickness=2):
        image_with_bbox = image.copy()

        x1, y1, x2, y2 = map(int, bbox.xyxy[0])
        cv2.rectangle(image_with_bbox, (x1, y1), (x2, y2), color, thickness)

        return image_with_bbox

    @staticmethod
    def get_label(box, names):
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = f"{names[cls]} {conf:.2f}"

        return label

    @staticmethod
    def draw_label(image, box, label, fontScale=0.5, color=(0, 255, 0), thickness=2):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        image_with_label = image.copy()
        
        cv2.putText(image_with_label, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)

        return image_with_label
    
    @staticmethod
    def resize_frame_to_fit(frame, max_width=640, max_height=480):
        h, w = frame.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def cv_image_to_qimage(img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return qimage