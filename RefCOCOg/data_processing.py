from PIL import Image
import pandas as pd
import numpy as np
import glob
import cv2
import io


def read_parquet(file_path):
    file_paths = glob.glob(file_path)
    dfs = [pd.read_parquet(file) for file in file_paths]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def bytes_to_image(image_bytes):
    return Image.open(io.BytesIO(image_bytes['bytes']))


def bytes2image(df):
    df['image'] = df['image'].apply(bytes_to_image)
    return df


def visualize_image(images):
    if isinstance(images, str):
        image = cv2.imread(images)
    elif isinstance(images, Image.Image):
        image = cv2.cvtColor(np.array(images), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("输入必须是字符串（文件路径）或 Pillow 图像对象")
    if image is None:
        raise ValueError("无法加载图像")

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segmentation_image(image, segmentation, bbox, question_answer):
    """
    可视化图像，并绘制 segmentation 和 bbox。
    :param image: 图像（可以是 Pillow 图像对象或 OpenCV 图像）
    :param segmentation: 分割区域（列表或 NumPy 数组）
    :param bbox: 边界框（列表，格式为 [x_min, y_min, width, height]
    """
    # 如果输入是 Pillow 图像对象，转换为 OpenCV 格式
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if isinstance(segmentation, list):
        segmentation = np.array(segmentation, dtype=np.int32)

    # 将 segmentation 转换为 (N, 2) 的点集
    segmentation = segmentation.reshape(-1, 2).astype(np.int32)
    cv2.polylines(image, [segmentation], isClosed=True,
                  color=(0, 255, 0), thickness=2)

    # 绘制 bbox
    x_min, y_min, width, height = bbox
    x_max, y_max = x_min + width, y_min + height
    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max),
                  int(y_max)), color=(0, 0, 255), thickness=2)

    print(f"question is: {question_answer[0]}")
    print(f"answer is: {question_answer[1]}")

    cv2.imshow(f"image segmentation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_row(row):
    question_id = row['question_id']
    question = row['question']
    answer = row['answer']
    image = row['image']
    segmentation = row['segmentation']
    bbox = row['bbox']
    return {
        'question_id': question_id,
        'question': question,
        'answer': answer,
        'image': image,
        'segmentation': segmentation,
        'bbox': bbox
    }


def process_meta_data(df):
    meta_data = df.apply(process_row, axis=1)
    return meta_data


def get_item_by_idx(meta_data, idx):
    return meta_data[idx]


def get_items():
    pass


if __name__ == "__main__":
    df = read_parquet("RefCOCOg/data/test-00001-of-00002.parquet")
    df = bytes2image(df)
    df_top1 = df.head(1)
    print(df_top1)

    meta_data = process_meta_data(df)
    meta_data0 = get_item_by_idx(meta_data, 0)
    image, segmentation, bbox, qa = meta_data0.get('image'), meta_data0.get(
        'segmentation'), meta_data0.get('bbox'), (meta_data0.get('question'), meta_data0.get('answer'))

    visualize_image(image)
    segmentation_image(image, segmentation, bbox, qa)
