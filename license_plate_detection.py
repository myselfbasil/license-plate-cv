import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pytesseract
import os
import time
from PIL import Image

# Path to the pre-trained YOLOv8 model for license plate detection
MODEL_PATH = "exported_pretrained.pt"

# Configuration for OCR
pytesseract.pytesseract.tesseract_cmd = r'tesseract'  # Update this path to your Tesseract executable if needed

class LicensePlateDetector:
    def __init__(self, model_path, confidence_threshold=0.25):
        """
        Initialize the license plate detector with a pre-trained YOLOv8 model.
        
        Args:
            model_path: Path to the YOLOv8 model
            confidence_threshold: Minimum confidence score to consider a detection valid
        """
        self.confidence_threshold = confidence_threshold
        # Load the model
        self.model = YOLO(model_path)
        print(f"Model loaded from {model_path}")
        
    def detect_license_plates(self, image):
        """
        Detect license plates in an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of detected license plate regions and their confidence scores
        """
        # Run inference with the model
        results = self.model(image)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence
                    })
        
        return detections
    
    def extract_license_plate_text(self, image, bbox):
        """
        Extract text from a license plate region using OCR.
        
        Args:
            image: Original image
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            Extracted text from the license plate
        """
        x1, y1, x2, y2 = bbox
        plate_img = image[y1:y2, x1:x2]
        
        # Preprocess the license plate image for better OCR results
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Use Tesseract to extract text
        config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(thresh, config=config).strip()
        
        return text
    
    def visualize_detections(self, image, detections, texts=None):
        """
        Draw bounding boxes and license plate text on the image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries with 'bbox' and 'confidence'
            texts: List of extracted license plate texts (optional)
            
        Returns:
            Image with visualized detections
        """
        img_copy = image.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)  # Increased thickness to 3
            
            # Draw text if available
            text = texts[i] if texts and i < len(texts) else ""
            info_text = f"{text} ({confidence:.2f})"
            
            # Calculate appropriate vertical offset for the text based on the increased font size
            vertical_offset = 50  # Larger offset to prevent text overlap with bounding box
            
            # Increased font size by 5 times (from 0.5 to 2.5)
            font_scale = 2.5
            # Increased thickness to make text more visible
            thickness = 6
            
            # Add a black background rectangle for better text visibility
            (text_width, text_height), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(img_copy, 
                        (bbox[0], bbox[1] - vertical_offset - text_height), 
                        (bbox[0] + text_width, bbox[1] - vertical_offset + 5), 
                        (0, 0, 0), 
                        -1)  # Filled rectangle
            
            # Put text on the image with larger font size
            cv2.putText(img_copy, info_text, 
                      (bbox[0], bbox[1] - vertical_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      font_scale, (0, 255, 0), thickness)
            
        return img_copy

    def process_image(self, image):
        """
        Process a single image for license plate detection and recognition.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Processed image with visualized results and a list of detected license plates
        """
        # Detect license plates
        detections = self.detect_license_plates(image)
        
        # Extract text from each detected plate
        license_texts = []
        for detection in detections:
            text = self.extract_license_plate_text(image, detection['bbox'])
            license_texts.append(text)
        
        # Visualize results
        output_image = self.visualize_detections(image, detections, license_texts)
        
        return output_image, license_texts, detections
    
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process a video for license plate detection and recognition.
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the output video (optional)
            display: Whether to display the video during processing
            
        Returns:
            List of detected license plates throughout the video
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_plates = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Process every 3rd frame to speed up processing (adjustable)
            if frame_count % 3 != 0 and frame_count > 1:
                if display:
                    cv2.imshow('License Plate Detection', frame)
                if writer:
                    writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Process the frame
            processed_frame, texts, detections = self.process_image(frame)
            
            # Store detected license plates
            for i, text in enumerate(texts):
                if text:  # Only add non-empty texts
                    plate_info = {
                        'frame': frame_count,
                        'text': text,
                        'confidence': detections[i]['confidence'],
                        'bbox': detections[i]['bbox']
                    }
                    all_plates.append(plate_info)
            
            # Display the processed frame
            if display:
                cv2.imshow('License Plate Detection', processed_frame)
                
            # Write to output video
            if writer:
                writer.write(processed_frame)
                
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
            
        return all_plates

    def process_webcam(self, camera_id=0, output_path=None):
        """
        Process webcam feed for license plate detection and recognition.
        
        Args:
            camera_id: Camera device ID (default: 0)
            output_path: Path to save the output video (optional)
            
        Returns:
            List of detected license plates
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open webcam with ID {camera_id}")
            return []
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20  # Approximate FPS for webcam
        
        # Create video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_plates = []
        frame_count = 0
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Process every 5th frame to maintain real-time performance
            if frame_count % 5 != 0 and frame_count > 1:
                cv2.imshow('License Plate Detection (Webcam)', frame)
                if writer:
                    writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Process the frame
            processed_frame, texts, detections = self.process_image(frame)
            
            # Store detected license plates
            for i, text in enumerate(texts):
                if text:  # Only add non-empty texts
                    plate_info = {
                        'frame': frame_count,
                        'text': text,
                        'confidence': detections[i]['confidence'],
                        'bbox': detections[i]['bbox']
                    }
                    all_plates.append(plate_info)
            
            # Display the processed frame
            cv2.imshow('License Plate Detection (Webcam)', processed_frame)
                
            # Write to output video
            if writer:
                writer.write(processed_frame)
                
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
            
        return all_plates


def main():
    # Initialize the license plate detector
    detector = LicensePlateDetector(MODEL_PATH)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='License Plate Detection and Recognition')
    parser.add_argument('--input', type=str, help='Path to input image or video file')
    parser.add_argument('--output', type=str, default=None, help='Path to save output file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam as input')
    parser.add_argument('--display', action='store_true', default=True, help='Display the results')
    args = parser.parse_args()
    
    # Check if a valid input is provided
    if not args.webcam and not args.input:
        parser.error("Please provide either --input or --webcam option")
    
    # Process based on the input type
    if args.webcam:
        print("Starting license plate detection with webcam...")
        plates = detector.process_webcam(output_path=args.output)
    elif args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Process image
        print(f"Processing image: {args.input}")
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not read image {args.input}")
            return
        
        processed_image, texts, _ = detector.process_image(image)
        
        # Display the results
        if args.display:
            cv2.imshow('License Plate Detection', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Save the output image
        if args.output:
            cv2.imwrite(args.output, processed_image)
            print(f"Output saved to {args.output}")
        
        # Print detected license plates
        print("Detected license plates:")
        for text in texts:
            if text:
                print(f"  - {text}")
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Process video
        print(f"Processing video: {args.input}")
        plates = detector.process_video(args.input, args.output, args.display)
        
        # Print detected license plates
        unique_plates = {}
        for plate in plates:
            text = plate['text']
            if text not in unique_plates:
                unique_plates[text] = plate
        
        print("\nDetected license plates:")
        for text, plate in unique_plates.items():
            print(f"  - {text} (confidence: {plate['confidence']:.2f})")
    else:
        print(f"Unsupported file format: {args.input}")
        return

if __name__ == "__main__":
    main()
