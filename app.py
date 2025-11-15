import cv2
import supervision as sv
from ultralytics import YOLO
import os

def main():
    # Full path to screw model
    model_path = r"E:\Documents\GitHub\mldetection_jetson\models\model.onnx"
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print(f"Current directory: {os.getcwd()}")
        return
    
    # Load the YOLO model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print("Model loaded successfully!")
    
    # Open webcam (camera ID 0 is the default webcam)
    camera_id = 0
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    print("Camera opened successfully!")
    
    # Create annotators for visualization
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    print("Starting detection... Press 'q' to quit")
    
    # Main detection loop
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Run detection on the frame
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Count cards with good confidence in this frame
        card_count = 0
        if len(detections) > 0 and detections.confidence is not None:
            # Only count detections with reasonable confidence
            card_count = sum(1 for conf in detections.confidence if conf > 0.5)
        
        # Draw bounding boxes and labels
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
        
        # Add card count to the frame
        cv2.putText(
            annotated_frame, 
            f"Objects in view: {card_count}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Show the result
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