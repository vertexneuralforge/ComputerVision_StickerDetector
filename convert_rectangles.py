import json
import os
from pathlib import Path

CLASSES = {"happy": 0, "sad": 1, "dead": 2}

def convert():
    base = Path(__file__).parent
    json_dir = base / "labelme_jsons"
    output_dir = base / "labels"
    
    output_dir.mkdir(exist_ok=True)
    
    for json_file in json_dir.glob("*.json"):
        print(f"\nProcessing {json_file.name}...")
        try:
            with open(json_file) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ERROR: Failed to load JSON - {e}")
            continue
        
        if "shapes" not in data:
            print("  ERROR: No 'shapes' key in JSON")
            continue
            
        txt_lines = []
        for shape in data.get("shapes", []):
            # Verify rectangle
            if shape.get("shape_type") != "rectangle":
                print(f"  Skipping non-rectangle shape in {json_file.name}")
                continue
                
            label = shape.get("label", "").lower()
            if label not in CLASSES:
                print(f"  ERROR: Unknown label '{label}'")
                continue
                
            points = shape.get("points", [])
            if len(points) != 2:
                print(f"  ERROR: Need 2 points for rectangle (got {len(points)})")
                continue
                
            # Calculate YOLO bbox
            x1, y1 = points[0]
            x2, y2 = points[1]
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            center_x = (x1 + width/2) / data["imageWidth"]
            center_y = (y1 + height/2) / data["imageHeight"]
            width /= data["imageWidth"]
            height /= data["imageHeight"]
            
            txt_lines.append(f"{CLASSES[label]} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        # Save
        output_path = output_dir / f"{json_file.stem}.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(txt_lines))
        print(f"  Saved {len(txt_lines)} boxes to {output_path.name}")

if __name__ == "__main__":
    convert()
    print("\nDone! Check the 'labels' folder.")