import cv2
import supervision as sv
from ultralytics import YOLO
import os
import time

def main():
    # Relative path to model
    model_path = os.path.join("models", "best_f16.engine")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print(f"Current directory: {os.getcwd()}")
        if os.path.exists('models'):
            print(f"Files in models/: {os.listdir('models')}")
        return
    
    # Load the YOLO model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print("Model loaded successfully!")
    
    # Open webcam
    camera_id = 0
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    print("Camera opened successfully!")
    
    # Create annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    print("Starting detection... Press 'q' to quit")
    
    # Main detection loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Run detection with timing
        start = time.time()
        results = model(frame)[0]
        inference_time = (time.time() - start) * 1000
        print(f"Inference: {inference_time:.1f}ms")
        
        detections = sv.Detections.from_ultralytics(results)
        
        # Count objects with confidence > 0.5
        object_count = 0
        if len(detections) > 0 and detections.confidence is not None:
            object_count = sum(1 for conf in detections.confidence if conf > 0.5)
        
        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
        
        # Add object count
        cv2.putText(
            annotated_frame, 
            f"Objects in view: {object_count} | Inference: {inference_time:.1f}ms", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Show result
        cv2.imshow("Object Detection", annotated_frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
