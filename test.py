import os
from ultralytics import YOLO
import cv2

def run_detection(test_dir, result_dir):
    # ====== Added Error Handling ======
    if not os.path.exists('best.pt'):
        raise FileNotFoundError("Missing 'best.pt'. Train the model first with train.py!")
    
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    # =================================

    model = YOLO('best.pt')
    os.makedirs(result_dir, exist_ok=True)

    for img_name in os.listdir(test_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_dir, img_name)
            results = model(img_path)
            
            # Save visualized image
            res_img = results[0].plot()
            output_path = os.path.join(result_dir, img_name)
            cv2.imwrite(output_path, res_img)
            
            # Print bounding boxes
            for box in results[0].boxes:
                xyxy = box.xyxy[0].tolist()
                cls = model.names[int(box.cls)]
                print(f"{img_name} ({int(xyxy[0])},{int(xyxy[1])})-({int(xyxy[2])},{int(xyxy[3])}) {cls}")

if __name__ == "__main__":
    import sys
    test_dir = sys.argv[1]
    result_dir = sys.argv[2]
    run_detection(test_dir, result_dir)