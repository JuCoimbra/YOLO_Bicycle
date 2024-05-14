import os
from datetime import datetime

import cv2
import tqdm

from src.app.services.detection_service import DetectionService
from src.app.settigs import Settings


def main():
    print("Started the application...")
    net = DetectionService.load_yolo()
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Carregar o vídeo
    cap = cv2.VideoCapture(Settings.videoPath)
    if not cap.isOpened():
        print("Cannot open video")
        exit()

    # Inicializar o rastreador de objetos
    tracked = cv2.TrackerCSRT.create()

    # Inicializar a lista de objetos detectados
    detected_objects: list = []

    # Vídeo de saida
    codec = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(
        os.path.join(
            Settings.trackedVideoFolderPath,
            f"tracked_video_{datetime.now().timestamp()}.avi",
        ),
        codec,
        30,
        (int(cap.get(3)), int(cap.get(4))),
    )

    frame_count = Settings.max_frames or cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for _ in tqdm.tqdm(range(int(frame_count))):
        # Lê um frame do vídeo. Ret representa o sucesso ou falha da leitura
        ret, frame = cap.read()

        # Verifica se foi possível ler o vídeo
        if not ret:
            print("Ret error")
            break

        # Atualizar a posição dos objetos com o tracking. Caso a lista esteja vazia, ele não faz nada
        DetectionService.update_objects_box(frame, detected_objects, tracked)

        # Lista das caixas delimitadoras (x, y, w, h) das bicicletas identificadas
        detected_objects = DetectionService.detect_objects(
            frame, net, output_layers, output_video
        )

        # Comparar detecções com objetos rastreados
        for object_det in detected_objects:

            # Se a lista estiver vazia, adiciona e inicia o tracking
            if not detected_objects:
                detected_objects.append(object_det)
                bbox_obj = [round(num) for num in object_det[1]]
                tracked.init(frame, tuple(bbox_obj))
                continue

            # Comparar o objeto detectado com as bicicletas já contadas
            matched = False
            for obj in detected_objects:
                overlap = DetectionService.calculate_overlap(object_det[1], obj[1])

                # Caso a semelhança seja superior a 60%, o objeto não é adicionado a lista de bicicletas (objects)
                if overlap > 0.6:
                    matched = True
                    break

            # Se a detecção não corresponder a nenhum objeto rastreado, adiciona à lista de bicicletas (objects)
            if not matched:
                detected_objects.append(object_det)

        # TODO: essa condição deveria estar dentro de uma função com nome autoexplicativo
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Objectos detectados: {len(detected_objects)}")
    cap.release()
    cv2.destroyAllWindows()
    output_video.release()
