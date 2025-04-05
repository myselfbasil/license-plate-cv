# License Plate Detection and Visualization

This project uses a pre-trained YOLOv8 model to detect license plates in images and videos, extract the text using OCR, and visualize the results by displaying the license plate numbers on the video playback.

## Features

- License plate detection using YOLOv8
- Text extraction from license plates using Tesseract OCR
- Real-time processing with webcam support
- Video file processing with visualization
- Image processing

## Installation

1. Make sure you have Python 3.8+ installed

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - On Windows: Download and install from [here](https://github.com/UB-Mannheim/tesseract/wiki)
   - On macOS: `brew install tesseract`
   - On Ubuntu: `sudo apt install tesseract-ocr`

4. Update the path to Tesseract in the script if needed (the `pytesseract.pytesseract.tesseract_cmd` line)

5. Test with your own video file, or else download one sample from [here](https://drive.google.com/file/d/1zGq5uX3NojJ2uFNra9auUD2x7R72ym38/view?usp=sharing)

## Usage

### Processing an image:
```
python license_plate_detection.py --input path/to/image.jpg --output path/to/output.jpg
```

### Processing a video:
```
python license_plate_detection.py --input path/to/video.mp4 --output path/to/output.mp4
```

### Using webcam:
```
python license_plate_detection.py --webcam --output path/to/output.mp4
```

### Options:
- `--input`: Path to input image or video file
- `--output`: Path to save the output file (optional)
- `--webcam`: Use webcam as input
- `--display`: Display the results (default: True)

## How It Works

1. The YOLOv8 model detects license plates in each frame
2. Each detected license plate region is cropped from the image
3. The cropped region is preprocessed for better OCR results
4. Tesseract OCR extracts the text from the license plate
5. The results are visualized on the original frame
6. Processed frames are displayed and/or saved to an output file

## Model

This project uses a pre-trained YOLOv8 model specifically trained for license plate detection: `exported_pretrained.pt`

## License

This project is provided under the MIT License.
