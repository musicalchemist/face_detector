import cv2
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import argparse
import os
import glob

# Create face detector
mtcnn = MTCNN()

def detect_faces(image_path, save_path, threshold):
    if not os.path.isfile(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return None

    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: The image file '{image_path}' could not be read.")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, scores = mtcnn.detect(img_rgb)

    if boxes is not None:
        for box, score in zip(boxes, scores):
            # Ignore detections with confidence score lower than 0.95
            if score < threshold:
                continue

            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green color and thickness is 3

    cv2.imwrite(save_path, img)

    return img


def display_image(image_path, threshold):
    image_path = os.path.abspath(image_path)
    save_path = os.path.abspath("./data/processed/" + os.path.basename(image_path))
    img = detect_faces(image_path, save_path, threshold)

    if img is None:
        print(f"Error: Unable to process the image '{image_path}'.")
        return

    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        print(f"Error: Unable to convert the image from BGR to RGB. {e}")
        return

    plt.imshow(img_rgb)
    plt.show()

def process_batch(folder_path, threshold):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    for image_file in image_files:
        image_file = os.path.abspath(image_file)
        save_path = os.path.abspath("./data/processed/" + os.path.basename(image_file))
        detect_faces(image_file, save_path, threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face detection script.')
    parser.add_argument('mode', type=str, choices=['manual', 'batch'], help='Mode of operation: "manual" for single image or "batch" for a folder of images.')
    parser.add_argument('path', type=str, help='Path to the image or folder where faces will be detected.')
    parser.add_argument('--threshold', type=float, default=0.95, help='Confidence threshold for face detection. Defaults to 0.95.')
    args = parser.parse_args()

    if args.mode == 'manual':
        display_image(args.path, args.threshold)
    else: # batch mode
        process_batch(args.path, args.threshold)

