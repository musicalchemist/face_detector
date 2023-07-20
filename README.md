# Face Identifier

## About

Face Identifier is a Python-based project that identifies faces in images. It uses the Multi-task Cascaded Convolutional Networks (MTCNN) model for face detection, and OpenCV for image processing.

This project can be run in two modes: manual or batch. In the manual mode, the script processes one image at a time, while in batch mode, it processes all images in a specified folder.

## Requirements

- Python 3.10 or later
- OpenCV
- Matplotlib
- MTCNN
- Numpy
- PyTorch
- Torchvision

You can install these requirements using pip:
`pip install opencv-python numpy matplotlib facenet-pytorch torch torchvision`

## Installation

1. Clone this repository to your local machine using `git clone`.
2. Navigate into the project directory: `cd face_identifier`.
3. It is recommended to create a virtual environment to keep the dependencies required by this project separate from your global Python environment. You can do this with the following command:

```shell
conda create --name face_detector python=3.10
```

4. Activate the virtual environment:

```shell
conda activate face_detector
```

5. Install the necessary requirements: `pip install -r requirements.txt`.

## Usage

Before running the script, make sure to navigate into the `src` directory:

```shell
cd src

```

To run the script in manual mode with a single image:

```shell
python script.py manual data/raw/example_1.jpg
```

To run the script in batch mode with a directory:

```shell
python script.py batch data/raw/
```

Both commands will process the images and save the output images with detected faces in the data/processed/ directory.

Additionally, you can specify a confidence threshold for face detection:

```shell
python script.py manual data/raw/example_1.jpg --threshold 0.90
```

If not specified, the confidence threshold defaults to 0.95.
