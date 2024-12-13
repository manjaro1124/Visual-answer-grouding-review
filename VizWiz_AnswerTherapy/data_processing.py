from enum import Enum
import os
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image


class Colors(Enum):
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    GRAY = (128, 128, 128)


colors = [
    Colors.GREEN.value,
    Colors.RED.value,
    Colors.BLUE.value,
    Colors.YELLOW.value,
    Colors.CYAN.value,
    Colors.MAGENTA.value,
    Colors.GRAY.value
]


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    if not isinstance(raw_data, dict):
        raise ValueError("raw_data 必须是字典类型")
    return raw_data


def build_meta_data(raw_data, image_dir):
    meta_data = []
    for image_name, items in raw_data.items():
        each_data = {}
        image_path = os.path.join(image_dir, image_name)
        each_data["image_path"] = image_path
        each_data["answers"] = items['answers']
        each_data["question"] = items['question']
        each_data["answer_type"] = items['answer_type']
        each_data["worker0"] = items['workers_results'].get('worker0')
        each_data["worker1"] = items['workers_results'].get('worker1')
        meta_data.append(each_data)
    return meta_data


def visualize_image(image, bboxes):
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError("输入必须是字符串（文件路径）、Pillow 图像对象或 numpy 数组")

    if image is None:
        raise ValueError("无法加载图像，请检查输入路径或数据是否正确")

    for i, bbox in enumerate(bboxes):
        if not isinstance(bbox, list):
            print("相关信息为:", bbox)
            continue
        segmentation = np.array([[point['x'], point['y']]
                                for point in bbox], dtype=np.int32)
        color = colors[i % len(colors)]
        cv2.polylines(image, [segmentation], isClosed=True,
                      color=color, thickness=2)

    cv2.imshow("Image with BBoxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    raw_data = read_json(
        "VizWiz_AnswerTherapy/data/Metadata/Metadata/VizWiz_val_AnswerTherapy_metadata.json"
    )
    image_dir = "VizWiz_AnswerTherapy/data/val/val"
    meta_data = build_meta_data(raw_data, image_dir)
    print("Total dataset size: ", len(meta_data))

    for i in range(10,20):
        single_data = meta_data[i]
        image_path, question, answers, worker0, worker1 = single_data['image_path'], single_data[
            'question'], single_data['answers'], single_data['worker0'], single_data['worker1']

        print("question", question)
        print("answers", answers)

        visualize_image(image_path, worker0)
        visualize_image(image_path, worker1)
