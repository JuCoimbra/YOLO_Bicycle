import gdown
import cv2
from git import Repo
import numpy as np
from dotenv import load_dotenv
from os import path, makedirs, environ

# Carregando as variáveis de ambiente do arquivo .env
load_dotenv()
yoloPath = environ["YOLO_PATH"]
weightsPath = environ["YOLO_WEIGHTS"]
modelPath = environ["YOLO_MODEL"]
classesPath = environ["YOLO_CLASSES"]
gitYPath = environ["GITY_PATH"]
videoPath = environ["VIDEO"]
trackedVideoPath = environ["TRACKED_VIDEO"]


def configureYOLO():
    if not path.exists(yoloPath ):
        makedirs(path.dirname(yoloPath))

    if not path.exists(weightsPath):
        # Baixar os pesos pré-treinados
        gdown.download('https://pjreddie.com/media/files/yolov3.weights', weightsPath, quiet=False)

    if not path.exists(gitYPath):
        Repo.clone_from('https://github.com/pjreddie/darknet.git', gitYPath)

configureYOLO()

# Carregar o modelo YOLO
net = cv2.dnn.readNet(weightsPath, modelPath)

# Carregar as classes
classes = open(classesPath, "r").read().strip().split('\n')

# Gera cores aleatórias, mas garante que as próximas operações de geração de números aleatórios produzirão os mesmos resultados em diferentes execuções do código
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Carregar o vídeo
cap = cv2.VideoCapture(videoPath)
if not cap.isOpened():
 print("Cannot open video")
 exit()

# Inicializar o rastreador de objetos
tracked = cv2.TrackerCSRT.create()

# Inicializar a lista de objetos detectados
objects = []

# Vídeo de saida
codec = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(trackedVideoPath, codec, 30, (int(cap.get(3)), int(cap.get(4))))



def update_objects_box(frame):
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

      draw_bounding_box(frame, class_id, round(x), round(y), round(x + w), round(y + h))

# Traça um retangulo em volta do objeto detectado
def draw_bounding_box(img, class_id, x, y, xw, yh):

    label = str(classes[class_id])

    color = colors[class_id]

    cv2.rectangle(img, (x, y), (xw, yh), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def detect_objects(frame):
    detectedBikes = []
    he, wi = frame.shape[:2]
    class_ids = []
    confidences = []
    boxes = []

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    outs = net.forward(output_layers)
    outputs = np.vstack(outs)

    for out in outs:
      for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence >= 0.95 and classes[class_id] == "bicycle":
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
      draw_bounding_box(frame, class_ids[i], round(x), round(y), round(x+w), round(y+h))

    output_video.write(frame)
    return detectedBikes

def calculate_overlap(box1, box2):
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


if __name__ == "__main__":
   while True:
    # Lê um frame do vídeo. Ret representa o sucesso ou falha da leitura
    ret, frame = cap.read()

    # Verifica se foi possível ler o vídeo
    if not ret:
      print('Ret error')
      break

    # Atualizar a posição dos objetos com o tracking. Caso a lista esteja vazia, ele não faz nada
    update_objects_box(frame)

    # Lista das caixas delimitadoras (x, y, w, h) das bicicletas identificadas
    detected_objects = detect_objects(frame)

    # Comparar detecções com objetos rastreados
    for object_det in detected_objects:

      # Se a lista estiver vazia, adiciona e inicia o tracking
      if not objects:
        objects.append(object_det)
        bbox_obj = [round(num) for num in object_det[1]]
        tracked.init(frame, tuple(bbox_obj))
        continue

      # Comparar o objeto detectado com as bicicletas já contadas
      matched = False
      for obj in objects:
        # Calcula a Intersection over Union dos objetos
        overlap = calculate_overlap(object_det[1], obj[1])

        # Caso a semelhança seja superior a 60%, o objeto não é adicionado a lista de bicicletas (objects)
        if overlap > 0.6:
          matched = True
          break

      # Se a detecção não corresponder a nenhum objeto rastreado, adiciona à lista de bicicletas (objects)
      if not matched:
        objects.append(object_det)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


print(f'Objectos detectados: {len(objects)}')
cap.release()
cv2.destroyAllWindows()
output_video.release()


