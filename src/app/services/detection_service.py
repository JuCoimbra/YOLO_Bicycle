import cv2
import numpy as np

from src.app.settigs import Settings

# Carregar as classes
CLASSES = open(Settings.classesPath, "r").read().strip().split("\n")

# Gera cores aleatórias, mas garante que as próximas operações de geração de números aleatórios produzirão os mesmos resultados em diferentes execuções do código
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class DetectionService:
    @staticmethod
    def update_objects_box(frame, objects, tracked):
        for obj in objects:
            if frame is None:
                continue

            if not tracked:
                print("O rastreador não está inicializado corretamente.")
                return

            detection = obj[0]
            ret, box = tracked.update(frame)

            if ret:
                x, y, w, h = [int(v) for v in box]

                scores = detection[5:]
                class_id = np.argmax(scores)

                DetectionService.draw_bounding_box(
                    frame, class_id, round(x), round(y), round(x + w), round(y + h)
                )


    @staticmethod# Traça um retangulo em volta do objeto detectado
    def draw_bounding_box(img, class_id, x, y, xw, yh):
        label = str(CLASSES[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x, y), (xw, yh), color, 2)

        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    @staticmethod
    def detect_objects(frame, net, output_layers, output_video):
        detectedBikes = []
        he, wi = frame.shape[:2]
        class_ids = []
        confidences = []
        boxes = []

        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        net.setInput(blob)

        outs = net.forward(output_layers)
        outputs = np.vstack(outs)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence >= 0.95 and CLASSES[class_id] == "bicycle":
                    center_x = int(detection[0] * wi)
                    center_y = int(detection[1] * he)
                    w = int(detection[2] * wi)
                    h = int(detection[3] * he)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    detectedBikes.append((detection, [x, y, w, h]))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for j in indices:
            if type(j) == list:
                i = j[0]
            else:
                i = j
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            DetectionService.draw_bounding_box(
                frame, class_ids[i], round(x), round(y), round(x + w), round(y + h)
            )

        output_video.write(frame)
        return detectedBikes

    @staticmethod
    def calculate_overlap(box1, box2):
        """
            Calcula a Intersection over Union (IoU) dos objetos.
        """
        # Extrair as coordenadas das caixas delimitadoras
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calcular as coordenadas dos pontos de interseção
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(w1, w2)
        y_bottom = min(h1, h2)

        # Calcular a área da interseção
        intersection_area = abs(max(x_right - x_left, 0) * max(y_bottom - y_top, 0))

        # Se o objeto não tiver area de interseção, retorna 0 imediatamente
        if intersection_area == 0:
            return 0.0

        # Calcular a área da união
        box1_area = abs(w1 - x1) * abs(h1 - y1)
        box2_area = abs(w2 - x2) * abs(h2 - y2)
        union_area = float(box1_area + box2_area - intersection_area)

        # Calcular o IoU - Intersection over Union
        iou = intersection_area / float(union_area)

        return iou
    
    @staticmethod
    def load_yolo():
        return cv2.dnn.readNet(Settings.weightsPath, Settings.modelPath)
