import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sff import SFFReader
from boxmot import DeepOcSort
from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# print(class_list)
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="yolov11",
    model_path="yolo11x.pt",        
    confidence_threshold=0.3,
    device="cuda:0",
)

class_list = {int(k): v for k, v in sahi_model.category_mapping.items()}
print(class_list)
target_classes = [2, 3, 5, 7]

tracker = DeepOcSort(
    reid_weights=Path('weights/osnet_ain_x1_0_msmt17.pt'),
    device=0,
    half=True,
    det_thresh=0.3,
    max_age=30,
    min_hits=2,
    iou_threshold=0.3,
    delta_t=3,
    inertia=0.4, #0.3-0.4
    w_association_emb=0.9, #0.6-0.8
    alpha_fixed_emb=0.9,
    cmc_off=True,
    per_class=True,


)

output_folder = "deep_oc_sort_videos"
# sff_files = [
#     r'D:\учеба\Практика\расчет_интенсивности\мониторинг_видео\Нефтеюганск_2023_ул.Набережная-Ленина\Camera_1\Video.sff',
#     r'D:\учеба\Практика\расчет_интенсивности\мониторинг_видео\Нефтеюганск_2023_ул.Набережная-Ленина\Camera_2\Video.sff',
#     r'D:\учеба\Практика\расчет_интенсивности\мониторинг_видео\Нефтеюганск_2023_ул.Набережная-Ленина\Camera_3\Video.sff'
# ]
# readers = [SFFReader(path) for path in sff_files]
# generators = [reader.get_frames() for reader in readers]

# print(readers[1].get_frames())

sff_file = r'D:\учеба\Практика\расчет_интенсивности\мониторинг_видео\Нефтеюганск_2023_ул.Набережная-Ленина\Camera_2\Video.sff'
reader = SFFReader(sff_file=sff_file)
scale = 2560 / (3 * 2048)  # ≈ 0.416
target_size = (int(2048 * scale), int(1536 * scale))  # (854, 639)
fps = 1000
delay_ms = int(1000 / fps)  

pts_red_12 = np.array([
    [400, 120],  # Точка 1 (верхний левый)
    [606, 120],  # Точка 2 (верхний правый)
    [606, 237],  # Точка 3 (нижний правый)
    [404, 233]    # Точка 4 (нижний левый)
], np.int32)

pts_blue_12 = np.array([
    [570, 210],  # Точка 1 (верхний левый)
    [845, 211],  # Точка 2 (верхний правый)
    [847, 350],  # Точка 3 (нижний правый)
    [570, 350]    # Точка 4 (нижний левый)
], np.int32)

pts_blue_12 = pts_blue_12.reshape((-1, 1, 2))
pts_red_12 = pts_red_12.reshape((-1, 1, 2))

pts_red_7 = np.array([
    [200, 195],  # Точка 1 (верхний левый)
    [380, 195],  # Точка 2 (верхний правый)
    [380, 240],  # Точка 3 (нижний правый)
    [200, 240]    # Точка 4 (нижний левый)
], np.int32)


pts_blue_7 = np.array([
    [480, 280],  # Точка 1 (верхний левый)
    [720, 300],  # Точка 2 (верхний правый)
    [514, 633],  # Точка 3 (нижний правый)
    [25, 500]    # Точка 4 (нижний левый)
], np.int32)

pts_blue_7 = pts_blue_7.reshape((-1, 1, 2))
pts_red_7 = pts_red_7.reshape((-1, 1, 2))
 
counted_ids_7 = set()
counted_ids_12 = set()


count_7 = defaultdict(int) 
count_12 = defaultdict(int)  

track_id_class_history = defaultdict(list)

crossed_7 = {}
crossed_12 = {}

for frame_num, image, picket, timestamp in reader.get_frames():
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, target_size)
    results = get_sliced_prediction(
        image=frame,
        detection_model=sahi_model,
        slice_height=512,                # ваш image_size
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        perform_standard_pred=False,     # стандартный прогон по всему кадру не нужен
        postprocess_type="NMS",          # после срезов объединяем NMS’ом
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        postprocess_class_agnostic=False,
        verbose=0,                        # бесшумно
    )

    dets = []
    for obj in results.object_prediction_list:
        x1, y1, x2, y2 = obj.bbox.to_xyxy()
        cls_idx = obj.category.id
        conf = obj.score.value
        if cls_idx not in target_classes:
            continue
        dets.append([x1, y1, x2, y2, conf, cls_idx])


    if len(dets) > 0:
        detections = np.array(dets, dtype=np.float32)
    else:
        # если нет детекций, передаём пустой массив правильной формы
        detections = np.zeros((0, 6), dtype=np.float32)

    
    tracks = tracker.update(detections, frame)

    # Прозрачная отрисовка зон
    overlay = frame.copy()
    # 12 line
    cv2.fillPoly(overlay, [pts_red_12], color=(255, 192, 203))
    cv2.polylines(overlay, [pts_red_12], isClosed=True, color=(255, 192, 203), thickness=2)
    cv2.putText(overlay, '12 up', (410, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.fillPoly(overlay, [pts_blue_12], color=(255, 192, 203))
    cv2.polylines(overlay, [pts_blue_12], isClosed=True, color=(255, 192, 203), thickness=2)
    cv2.putText(overlay, '12 right', (690, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    #7 line
    cv2.fillPoly(overlay, [pts_red_7], color=(0, 165, 255))
    cv2.polylines(overlay, [pts_red_7], isClosed=True, color=(0, 165, 255), thickness=2)
    cv2.putText(overlay, '7 up', (260, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.fillPoly(overlay, [pts_blue_7], color=(0, 165, 255))
    cv2.polylines(overlay, [pts_blue_7], isClosed=True, color=(0, 165, 255), thickness=2)
    cv2.putText(overlay, '7 right', (35, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    for track in tracks:
        x1, y1, x2, y2, track_id, conf, class_id, _ = map(int, track)
        cx = (x1 + x2) // 2
        cy = y2
        class_name = class_list[class_id]
        track_id_class_history[track_id].append(class_name)
        track_id_class_history[track_id] = track_id_class_history[track_id][-15:]  # Можно увеличить
        class_name = Counter(track_id_class_history[track_id]).most_common(1)[0][0]

        # Визуализация трека
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # 12 line count
        if cv2.pointPolygonTest(pts_blue_12, (cx, cy), False) >= 0:
            if track_id not in crossed_12:
                crossed_12[track_id] = True
        
        if track_id in crossed_12 and track_id not in counted_ids_12:
            if cv2.pointPolygonTest(pts_red_12, (cx, cy), False) >= 0:
                if track_id in track_id_class_history and len(track_id_class_history[track_id]) >= 1:
                    top_class = Counter(track_id_class_history[track_id]).most_common(1)[0][0]
                    if class_list[class_id] != top_class:
                        class_id = class_list.index(top_class)
                    count_12[top_class] += 1
                    counted_ids_12.add(track_id)

        # 7 line count
        if cv2.pointPolygonTest(pts_red_7, (cx, cy), False) >= 0:
            if track_id not in crossed_7:
                crossed_7[track_id] = True
        
        if track_id in crossed_7 and track_id not in counted_ids_7:
            if cv2.pointPolygonTest(pts_blue_7, (cx, cy), False) >= 0:
                if track_id in track_id_class_history and len(track_id_class_history[track_id]) >= 1:
                    top_class = Counter(track_id_class_history[track_id]).most_common(1)[0][0]
                    if class_list[class_id] != top_class:
                        class_id = class_list.index(top_class)
                    count_7[top_class] += 1
                    counted_ids_7.add(track_id)

    y_offset = 30
    for class_name, count in count_7.items():
        cv2.putText(frame, f'{class_name} (7): {count}', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
        y_offset += 30

    y_offset += 20  
    for class_name, count in count_12.items():
        cv2.putText(frame, f'{class_name} (12): {count}', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 192, 203), 2, cv2.LINE_AA)
        y_offset += 30

    cv2.imshow("YOLO Tracking & Counting", frame)
    if cv2.waitKey(delay_ms) & 0xFF == 27:
        break

cv2.destroyAllWindows()