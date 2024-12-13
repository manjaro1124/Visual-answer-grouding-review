
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import json
import os


def read_json(file_path, mode='test'):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    if mode == 'train':
        print("loading training datasets")
        print(len(raw_data['train']))
        return raw_data['train']
    elif mode == 'val':
        print("loading val datasets")
        print(len(raw_data['val']))
        return raw_data['val']
    else:
        print("loading test datasets")
        print(len(raw_data['test']))
        return raw_data['test'][:10]


def visualize_image(image, bbox):
    """
    可视化图像，并绘制 bbox。
    :param image: 图像（可以是字符串路径、Pillow 图像对象或 numpy 数组）
    :param bbox: 边界框（字典，格式为 {'x': 中心点 x, 'y': 中心点 y, 'width': 宽度, 'height': 高度}）
    """
    # 判断输入类型
    if isinstance(image, str):
        # 如果输入是字符串（文件路径），使用 OpenCV 读取图像
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        # 如果输入是 Pillow 图像对象，转换为 OpenCV 格式
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError("输入必须是字符串（文件路径）、Pillow 图像对象或 numpy 数组")

    # 检查图像是否成功加载
    if image is None:
        raise ValueError("无法加载图像，请检查输入路径或数据是否正确")

    # 提取 bbox 的中心点、宽度和高度
    x_center, y_center = bbox['x'], bbox['y']
    width, height = bbox['width'], bbox['height']

    # 计算左上角和右下角的坐标
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    # 绘制 bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                  color=(0, 0, 255), thickness=2)

    # 显示图像
    cv2.imshow("Image with BBox", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_item_by_idx(meta_data, idx):
    return meta_data[idx]


def build_meta_data(raw_data, image_dir):
    meta_data = []
    for _, data in enumerate(raw_data):
        each_data = {}
        image_name = data['image_name']
        image_path = os.path.join(image_dir, image_name)
        each_data["knowledge"] = data['knowledge']
        each_data["ref_exp"] = data['ref_exp']
        each_data["bbox"] = data['bbox']
        each_data["image"] = cv2.imread(image_path)
        meta_data.append(each_data)
    return meta_data


if __name__ == "__main__":
    raw_data = read_json("SK-VG/data/annotations.json")
    image_dir = "SK-VG/data/images"
    meta_data = build_meta_data(raw_data, image_dir)
    meta_data0 = meta_data[6]
    image, bbox = meta_data0['image'], meta_data0['bbox']
    print("knowledge:", meta_data0['knowledge'])
    print("ref_exp:", meta_data0['ref_exp'])
    visualize_image(image, bbox)
