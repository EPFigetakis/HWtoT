import os
import json
import re
import pandas as pd 

def create_dataset_json(image_dir, label_dir, output_file="dataset.json"):
    dataset = []

    # Gather all image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    
    # Group images by prefix (001c, 002c, etc.)
    image_groups = {}
    pattern = re.compile(r"(\d{3})c_line_(\d+)\.png")

    for img in image_files:
        match = pattern.match(img)
        if not match:
            continue
        base_id = match.group(1)  # "001" from "001c"
        line_num = int(match.group(2))
        image_groups.setdefault(base_id, []).append((line_num, img))

    # Sort each group by line number
    for base_id in image_groups:
        image_groups[base_id] = sorted(image_groups[base_id], key=lambda x: x[0])

    # Match each group of images to corresponding label file (001.txt, etc.)
    for base_id, lines in image_groups.items():
        label_path = os.path.join(label_dir, f"{base_id}.txt")
        if not os.path.exists(label_path):
            print(f"⚠️ Missing label file for base_id: {base_id}")
            continue

        with open(label_path, 'r', encoding='utf-8') as f:
            label_lines = [line.strip() for line in f if line.strip()]

        if len(lines) != len(label_lines):
            print(f"⚠️ Line mismatch for {base_id}: {len(lines)} images vs {len(label_lines)} labels")

        for (line_num, img_name), label in zip(lines, label_lines):
            dataset.append({
                "image_path": os.path.join(image_dir, img_name),
                "label": label
            })

    # Save JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved {len(dataset)} items to {output_file}")

def convert_json_to_csv(json_path="dataset.json", csv_path="dataset.csv"):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ Converted to {csv_path} with {len(df)} entries.")