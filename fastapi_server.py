import asyncio
import base64
import glob
import json
import os
import re
from io import BytesIO

from fastapi import FastAPI, WebSocket
from PIL import Image
from tqdm import tqdm

from deepsort import DeepSortTracker
from hungarian import HungarianTracker
from track_15_10_75 import country_balls_amount, track_data

app = FastAPI(title="Tracker assignment")
imgs = glob.glob("imgs/*")
country_balls = [
    {"cb_id": x, "img": imgs[x % len(imgs)]} for x in range(country_balls_amount)
]
id_history = {id: [] for id in range(country_balls_amount)}
country_ball_images = {}
for ball in country_balls:
    cb_id = ball["cb_id"]
    img_path = ball["img"]
    try:
        country_ball_images[cb_id] = Image.open(img_path).convert("RGBA")
    except Exception as e:
        print(f"Failed to load image for cb_id {cb_id}: {e}")
dir = "save_frames_dir"
if not os.path.exists(dir):
    os.makedirs(dir)
print("Started")

hungarian_tracker = HungarianTracker(max_age=10, min_hits=3, distance_threshold=200)
deep_sort_tracker = DeepSortTracker(max_age=30, min_hits=3, iou_threshold=0.3)


def tracker_soft(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """
    el["data"] = hungarian_tracker.update(el["data"])
    track_ids = {}
    for det in el["data"]:
        if det["track_id"] is not None:
            if det["track_id"] in track_ids:
                det["track_id"] = None
            else:
                track_ids[det["track_id"]] = det
    return el


def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true и воспользуйтесь нижним закомментированным кодом в этом файле для первого прогона,
    на повторном прогоне можете читать сохраненные фреймы из папки
    и по координатам вырезать необходимые регионы.
    """
    frame_id = el["frame_id"] - 1
    el["data"] = deep_sort_tracker.update(frame_id, el["data"])
    track_ids = {}
    for det in el["data"]:
        if det["track_id"] is not None:
            if det["track_id"] in track_ids:
                det["track_id"] = None
            else:
                track_ids[det["track_id"]] = det
    return el


def calculate_tracker_metrics(id_history, track_data):
    id_switches_per_object = {}
    total_id_switches = 0
    for cb_id, track_ids in id_history.items():
        id_switches = 0
        last_valid_id = None
        for tid in track_ids:
            if tid is not None:
                if last_valid_id is not None and tid != last_valid_id:
                    id_switches += 1
                last_valid_id = tid
        id_switches_per_object[cb_id] = id_switches
        total_id_switches += id_switches
    tp_per_frame = []
    fn_per_frame = []
    for frame in track_data:
        tp = 0
        fn = 0
        for obj in frame["data"]:
            if obj["track_id"] is not None:
                tp += 1
            else:
                fn += 1
        tp_per_frame.append(tp)
        fn_per_frame.append(fn)
    total_tp = sum(tp_per_frame)
    total_fn = sum(fn_per_frame)
    total_gt = total_tp + total_fn
    if total_gt > 0:
        mota = 1 - (total_fn + total_id_switches) / total_gt
        tracking_accuracy = total_tp / total_gt
    else:
        mota = 0
        tracking_accuracy = 0
    results = {
        "MOTA": mota,
        "Tracking Accuracy": tracking_accuracy,
        "ID Switches": total_id_switches,
        "Total True Positives": total_tp,
        "Total False Negatives": total_fn,
        "Total Ground Truth": total_gt,
    }
    return results


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("Accepting client connection...")
    await websocket.accept()
    await websocket.receive_text()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте

    await websocket.send_text(str(country_balls))
    for el in tqdm(track_data):
        await asyncio.sleep(0.5)
        image_data = await websocket.receive_text()
        # print(image_data)
        try:
            image_data = re.sub("^data:image/.+;base64,", "", image_data)
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image = image.resize((1000, 800), Image.Resampling.LANCZOS)
            frame_id = el["frame_id"] - 1
            image.save(f"{dir}/frame_{frame_id}.png")
            # print(image)
        except Exception as e:
            print(e)

        # отправка информации по фрейму
        el = tracker_strong(el)
        for cb in el["data"]:
            id_history[cb["cb_id"]].append(cb["track_id"])
        await websocket.send_json(el)

    await websocket.send_json(el)
    await asyncio.sleep(0.5)
    image_data = await websocket.receive_text()
    try:
        image_data = re.sub("^data:image/.+;base64,", "", image_data)
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image = image.resize((1000, 800), Image.Resampling.LANCZOS)
        image.save(f"{dir}/frame_{el['frame_id']}.png")
    except Exception as e:
        print(e)

    print(id_history)
    metrics = calculate_tracker_metrics(id_history, track_data)
    print(metrics)
    with open("strong_metrics.jsonl", "a") as f:
        print(json.dumps(metrics), file=f)

    print("Bye..")
