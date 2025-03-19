# import os
# import json
# import glob
# import shutil
# import urllib.parse
# from sklearn.model_selection import train_test_split
# from ultralytics import YOLO

# # --------------------------
# # 1. 라벨 맵핑 (한글 라벨명 -> 클래스 ID)
# # --------------------------
# LABEL_MAP = {
#     "모델명": 0,
#     "사이즈": 1,
#     "DOT": 2
# }

# # --------------------------
# # 2. 확장자 무관 JSON 로드 함수
# # --------------------------
# def load_json_files_any_extension(json_dir):
#     """
#     json_dir 폴더 안의 모든 파일(확장자 무관)을 열어 JSON 파싱을 시도하고,
#     성공한 파일 경로들을 리스트로 반환합니다.
#     """
#     json_file_paths = []
#     for filename in os.listdir(json_dir):
#         file_path = os.path.join(json_dir, filename)
#         # 파일인지 확인
#         if os.path.isfile(file_path):
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     _ = json.load(f)  # JSON 파싱 시도
#                 json_file_paths.append(file_path)
#             except Exception as e:
#                 print(f"Warning: '{filename}' is not a valid JSON. Skipping. Error: {e}")
#     return json_file_paths

# # --------------------------
# # 3. Label Studio 이미지 경로 추출 함수
# # --------------------------
# def extract_image_path(image_field):
#     """
#     Label Studio의 image 필드 값 (예: "/data/local-files/?d=data/samples_tires/20250307_123258.jpeg")
#     에서 'd' 파라미터를 이용해 실제 이미지 경로를 추출합니다.
#     """
#     parsed = urllib.parse.urlparse(image_field)
#     qs = urllib.parse.parse_qs(parsed.query)
#     if "d" in qs:
#         return qs["d"][0]  # 예: data/samples_tires/20250307_123258.jpeg
#     else:
#         return image_field

# # --------------------------
# # 4. JSON -> YOLO 라벨 변환 함수
# # --------------------------
# def convert_annotation(json_path, temp_labels_dir):
#     """
#     하나의 JSON 파일을 읽어 YOLO annotation(.txt) 파일을 생성합니다.
#     Label Studio의 좌표값(x, y, width, height)은 백분율(% 단위)로 되어 있으므로,
#     YOLO 포맷 (class_id, center_x, center_y, width, height)를 0~1 범위 값으로 변환합니다.
    
#     생성된 라벨 파일은 temp_labels_dir에 저장되며,
#     반환값은 JSON 파일 내 이미지 경로(상대 경로)입니다.
#     """
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     if "task" not in data or "data" not in data["task"] or "image" not in data["task"]["data"]:
#         print(f"Warning: JSON '{json_path}' does not contain expected fields. Skipping.")
#         return None
    
#     image_field = data["task"]["data"]["image"]
#     image_file = extract_image_path(image_field)  # 예: data/samples_tires/xxx.jpeg
#     base_filename = os.path.splitext(os.path.basename(image_file))[0]
    
#     results = data.get("result", [])
#     if not results:
#         print(f"Warning: JSON '{json_path}' has no 'result' field. Skipping.")
#         return None
    
#     label_file_path = os.path.join(temp_labels_dir, base_filename + ".txt")
#     lines = []
    
#     for anno in results:
#         value = anno.get("value", {})
#         x = value.get("x", 0)
#         y = value.get("y", 0)
#         w = value.get("width", 0)
#         h = value.get("height", 0)
        
#         # YOLO는 중심 좌표 (cx, cy) + width, height 형태
#         center_x = (x + w/2) / 100.0
#         center_y = (y + h/2) / 100.0
#         width_norm = w / 100.0
#         height_norm = h / 100.0
        
#         labels = value.get("rectanglelabels", [])
#         if not labels:
#             continue
        
#         label_name = labels[0]
#         if label_name not in LABEL_MAP:
#             print(f"Warning: label '{label_name}' not in LABEL_MAP. Skipping.")
#             continue
        
#         class_id = LABEL_MAP[label_name]
#         line = f"{class_id} {center_x:.6f} {center_y:.6f} {width_norm:.6f} {height_norm:.6f}"
#         lines.append(line)
    
#     if not lines:
#         print(f"Warning: No valid annotations in '{json_path}'. Skipping.")
#         return None
    
#     with open(label_file_path, 'w', encoding='utf-8') as f:
#         for line in lines:
#             f.write(line + "\n")
    
#     return image_file

# # --------------------------
# # 5. 데이터셋 구성 함수
# # --------------------------
# def prepare_dataset(json_dir, images_dir, output_dataset_dir):
#     """
#     1) json_dir 폴더 내 모든 JSON 파일(확장자 무관) 파싱
#     2) YOLO 라벨(.txt) 파일 생성
#     3) train/val 분리 및 이미지, 라벨 파일 복사
#     4) YOLO 학습용 data.yaml 파일 생성
#     """
#     train_images_dir = os.path.join(output_dataset_dir, "train", "images")
#     train_labels_dir = os.path.join(output_dataset_dir, "train", "labels")
#     val_images_dir = os.path.join(output_dataset_dir, "val", "images")
#     val_labels_dir = os.path.join(output_dataset_dir, "val", "labels")
    
#     for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
#         os.makedirs(d, exist_ok=True)
    
#     temp_labels_dir = os.path.join(output_dataset_dir, "temp_labels")
#     os.makedirs(temp_labels_dir, exist_ok=True)
    
#     json_files = load_json_files_any_extension(json_dir)
#     if not json_files:
#         print(f"No JSON files found in '{json_dir}'")
    
#     dataset_entries = []  # (image_source_path, base_filename)
#     for json_file in json_files:
#         image_relative_path = convert_annotation(json_file, temp_labels_dir)
#         if not image_relative_path:
#             continue
        
#         image_basename = os.path.basename(image_relative_path)
#         image_source_path = os.path.join(images_dir, image_basename)
#         dataset_entries.append((image_source_path, image_basename))
    
#     n_samples = len(dataset_entries)
#     if n_samples == 0:
#         print("No valid dataset entries found. Aborting.")
#         return None
    
#     test_size = 0.2 if n_samples > 1 else 0.0
#     train_entries, val_entries = train_test_split(
#         dataset_entries,
#         test_size=test_size,
#         random_state=42
#     ) if n_samples > 1 else (dataset_entries, [])
    
#     def copy_entries(entries, dest_images_dir, dest_labels_dir):
#         for img_src, base_filename in entries:
#             if os.path.exists(img_src):
#                 shutil.copy(img_src, os.path.join(dest_images_dir, base_filename))
#             else:
#                 print(f"Warning: Image file not found: {img_src}")
            
#             label_filename = os.path.splitext(base_filename)[0] + ".txt"
#             src_label_path = os.path.join(temp_labels_dir, label_filename)
#             dest_label_path = os.path.join(dest_labels_dir, label_filename)
#             if os.path.exists(src_label_path):
#                 shutil.move(src_label_path, dest_label_path)
#             else:
#                 print(f"Warning: Label file not found: {src_label_path}")
    
#     copy_entries(train_entries, train_images_dir, train_labels_dir)
#     copy_entries(val_entries, val_images_dir, val_labels_dir)
    
#     shutil.rmtree(temp_labels_dir)
    
#     data_yaml_path = os.path.join(output_dataset_dir, "data.yaml")
#     data_yaml_content = f"""train: {os.path.abspath(train_images_dir)}
# val: {os.path.abspath(val_images_dir)}
# nc: {len(LABEL_MAP)}
# names: ["모델명", "사이즈", "DOT"]
# """
#     with open(data_yaml_path, 'w', encoding='utf-8') as f:
#         f.write(data_yaml_content)
    
#     print(f"[INFO] Dataset prepared. data.yaml saved at: {data_yaml_path}")
#     return data_yaml_path

# # --------------------------
# # 6. 메인 실행부
# # --------------------------
# if __name__ == "__main__":
#     # 경로 설정 (필요에 따라 수정)
#     json_dir = "./data/export"           # JSON 어노테이션 파일들이 있는 폴더
#     images_dir = "./data/samples_tires"    # 원본 이미지들이 있는 폴더
#     output_dataset_dir = "./yolo_dataset"  # 최종 데이터셋이 구성될 폴더
    
#     # 1) 데이터셋 준비
#     data_yaml_path = prepare_dataset(json_dir, images_dir, output_dataset_dir)
#     if not data_yaml_path:
#         print("Dataset preparation failed. Exiting.")
#         exit(1)
    
#     # 2) YOLOv8n 모델 학습 (사전학습 없이, from scratch)
#     #    weights 인자는 제거하고, 모델 생성 시 yolov8n.yaml을 사용하여 초기화
#     model = YOLO("yolov8n.yaml")
#     model.train(
#         data=data_yaml_path,
#         epochs=50,
#         imgsz=640,
#         name="yolov8n_scratch"
#     )

import os
import json
import glob
import shutil
import urllib.parse
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import cv2
import albumentations as A

#############################################
# 1. 기본 함수들 (JSON 파싱, 이미지 경로 추출, 라벨 변환)
#############################################

# 라벨 맵핑 (한글 라벨명 -> 클래스 ID)
LABEL_MAP = {
    "모델명": 0,
    "사이즈": 1,
    "DOT": 2
}

def load_json_files_any_extension(json_dir):
    """
    json_dir 폴더 내의 모든 파일(확장자 무관)을 JSON 파싱 시도 후
    정상적으로 파싱되는 파일 경로 리스트 반환
    """
    json_file_paths = []
    for filename in os.listdir(json_dir):
        file_path = os.path.join(json_dir, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    _ = json.load(f)
                json_file_paths.append(file_path)
            except Exception as e:
                print(f"Warning: '{filename}' is not a valid JSON. Skipping. Error: {e}")
    return json_file_paths

def extract_image_path(image_field):
    """
    Label Studio의 image 필드 (예: "/data/local-files/?d=data/samples_tires/20250307_123258.jpeg")
    에서 'd' 파라미터를 이용하여 실제 이미지 경로 추출
    """
    parsed = urllib.parse.urlparse(image_field)
    qs = urllib.parse.parse_qs(parsed.query)
    if "d" in qs:
        return qs["d"][0]
    else:
        return image_field

def convert_annotation(json_path, temp_labels_dir):
    """
    하나의 JSON 파일을 읽어 YOLO annotation (.txt) 파일로 변환하여 temp_labels_dir에 저장.
    좌표는 Label Studio의 % 단위(x,y,width,height)를 YOLO 형식 (center_x, center_y, width, height; 0~1)으로 변환.
    반환값: JSON 파일 내 이미지 경로(상대경로)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "task" not in data or "data" not in data["task"] or "image" not in data["task"]["data"]:
        print(f"Warning: JSON '{json_path}' does not contain expected fields. Skipping.")
        return None
    
    image_field = data["task"]["data"]["image"]
    image_file = extract_image_path(image_field)
    base_filename = os.path.splitext(os.path.basename(image_file))[0]
    
    results = data.get("result", [])
    if not results:
        print(f"Warning: JSON '{json_path}' has no 'result' field. Skipping.")
        return None
    
    label_file_path = os.path.join(temp_labels_dir, base_filename + ".txt")
    lines = []
    for anno in results:
        value = anno.get("value", {})
        x = value.get("x", 0)
        y = value.get("y", 0)
        w = value.get("width", 0)
        h = value.get("height", 0)
        center_x = (x + w/2) / 100.0
        center_y = (y + h/2) / 100.0
        width_norm = w / 100.0
        height_norm = h / 100.0
        labels = value.get("rectanglelabels", [])
        if not labels:
            continue
        label_name = labels[0]
        if label_name not in LABEL_MAP:
            print(f"Warning: label '{label_name}' not in LABEL_MAP. Skipping.")
            continue
        class_id = LABEL_MAP[label_name]
        line = f"{class_id} {center_x:.6f} {center_y:.6f} {width_norm:.6f} {height_norm:.6f}"
        lines.append(line)
    
    if not lines:
        print(f"Warning: No valid annotations in '{json_path}'. Skipping.")
        return None
    
    with open(label_file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + "\n")
    return image_file

#############################################
# 2. "all" 데이터셋 준비 (원본 JSON→YOLO 변환)
#############################################
def prepare_all_dataset(json_dir, images_dir, output_all_dir):
    """
    JSON 파일들을 파싱하여 원본 이미지와 YOLO 라벨(.txt)을
    output_all_dir (예: ./yolo_dataset/all) 에 복사
    """
    images_all_dir = os.path.join(output_all_dir, "images")
    labels_all_dir = os.path.join(output_all_dir, "labels")
    os.makedirs(images_all_dir, exist_ok=True)
    os.makedirs(labels_all_dir, exist_ok=True)
    
    temp_labels_dir = os.path.join(output_all_dir, "temp_labels")
    os.makedirs(temp_labels_dir, exist_ok=True)
    
    json_files = load_json_files_any_extension(json_dir)
    dataset_entries = []
    for json_file in json_files:
        image_relative_path = convert_annotation(json_file, temp_labels_dir)
        if not image_relative_path:
            continue
        image_basename = os.path.basename(image_relative_path)
        image_source_path = os.path.join(images_dir, image_basename)
        dataset_entries.append((image_source_path, image_basename))
    
    for img_src, base_filename in dataset_entries:
        if os.path.exists(img_src):
            shutil.copy(img_src, os.path.join(images_all_dir, base_filename))
        else:
            print(f"Warning: Image file not found: {img_src}")
        label_filename = os.path.splitext(base_filename)[0] + ".txt"
        src_label_path = os.path.join(temp_labels_dir, label_filename)
        dest_label_path = os.path.join(labels_all_dir, label_filename)
        if os.path.exists(src_label_path):
            shutil.move(src_label_path, dest_label_path)
        else:
            print(f"Warning: Label file not found: {src_label_path}")
    shutil.rmtree(temp_labels_dir)
    print(f"[INFO] All dataset prepared at: {output_all_dir}")
    return output_all_dir

#############################################
# 3. 회전 증강 설정 및 관련 함수
#############################################
# Albumentations 회전 증강 (±30도), YOLO 좌표 형식 사용
rotation_transform = A.Compose([
    A.Rotate(limit=180, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0))
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))

def read_yolo_label(label_path):
    """
    YOLO .txt 라벨을 읽어, boxes: [[cx,cy,w,h], ...]와
    class_ids: [class_id, ...] 리스트 반환
    """
    boxes = []
    class_ids = []
    if not os.path.exists(label_path):
        return boxes, class_ids
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            c, cx, cy, w, h = parts
            boxes.append([float(cx), float(cy), float(w), float(h)])
            class_ids.append(int(c))
    return boxes, class_ids

def write_yolo_label(label_path, boxes, class_ids):
    """
    boxes, class_ids를 YOLO .txt 형식으로 기록
    """
    with open(label_path, 'w', encoding='utf-8') as f:
        for box, cid in zip(boxes, class_ids):
            cx, cy, w, h = box
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def augment_all_data(input_images_dir, input_labels_dir, output_aug_images_dir, output_aug_labels_dir, num_aug=10):
    """
    input_images_dir, input_labels_dir에 있는 원본 10개 파일을 회전 증강하여
    num_aug배(예: 10배) 늘려 output_aug_images_dir, output_aug_labels_dir에 저장.
    """
    os.makedirs(output_aug_images_dir, exist_ok=True)
    os.makedirs(output_aug_labels_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_images_dir, "*.*"))
    for img_path in image_paths:
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(input_labels_dir, base_name + ".txt")
        if not os.path.exists(label_path):
            continue
        img = cv2.imread(img_path)
        boxes, class_ids = read_yolo_label(label_path)
        if len(boxes) == 0:
            continue
        for i in range(num_aug):
            transformed = rotation_transform(image=img, bboxes=boxes, class_labels=class_ids)
            aug_img = transformed['image']
            aug_boxes = transformed['bboxes']
            aug_class_ids = transformed['class_labels']
            aug_img_name = f"{base_name}_aug_{i}{ext}"
            aug_img_path = os.path.join(output_aug_images_dir, aug_img_name)
            cv2.imwrite(aug_img_path, aug_img)
            aug_label_name = f"{base_name}_aug_{i}.txt"
            aug_label_path = os.path.join(output_aug_labels_dir, aug_label_name)
            write_yolo_label(aug_label_path, aug_boxes, aug_class_ids)

#############################################
# 4. 증강된 데이터를 Train/Val로 분할하는 함수
#############################################
def split_dataset(aug_images_dir, aug_labels_dir, output_final_dir, test_size=0.2):
    """
    aug_images_dir, aug_labels_dir 내의 증강된 데이터를 읽어,
    output_final_dir 내에 train/ 및 val/ 폴더를 생성 (images, labels)
    """
    train_dir = os.path.join(output_final_dir, "train")
    val_dir = os.path.join(output_final_dir, "val")
    train_images_dir = os.path.join(train_dir, "images")
    train_labels_dir = os.path.join(train_dir, "labels")
    val_images_dir = os.path.join(val_dir, "images")
    val_labels_dir = os.path.join(val_dir, "labels")
    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(d, exist_ok=True)
    aug_image_paths = glob.glob(os.path.join(aug_images_dir, "*.*"))
    train_files, val_files = train_test_split(aug_image_paths, test_size=test_size, random_state=42)
    def copy_files(file_list, dst_images_dir, dst_labels_dir):
        for img_path in file_list:
            base_name = os.path.basename(img_path)
            label_name = os.path.splitext(base_name)[0] + ".txt"
            src_label_path = os.path.join(aug_labels_dir, label_name)
            shutil.copy(img_path, os.path.join(dst_images_dir, base_name))
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, os.path.join(dst_labels_dir, label_name))
            else:
                print(f"Warning: Label file not found for {img_path}")
    copy_files(train_files, train_images_dir, train_labels_dir)
    copy_files(val_files, val_images_dir, val_labels_dir)
    print("[INFO] Dataset splitting completed.")
    return train_images_dir, train_labels_dir, val_images_dir, val_labels_dir

#############################################
# 5. 메인 실행부: 전체 파이프라인 실행 및 모델 학습
#############################################
if __name__ == "__main__":
    # 경로 설정
    json_dir = "./data/export"            # JSON 어노테이션 파일들이 있는 폴더
    images_dir = "./data/samples_tires"     # 원본 이미지들이 있는 폴더
    
    base_dataset_dir = "./yolo_dataset"
    all_dir = os.path.join(base_dataset_dir, "all")           # 원본 데이터셋 (10개)
    aug_dir = os.path.join(base_dataset_dir, "augmented")       # 증강 데이터셋 (10배 → 100개)
    final_dir = os.path.join(base_dataset_dir, "final")         # 최종 Train/Val 데이터셋
    
    # 1. JSON → YOLO 데이터셋 변환 (모든 파일을 all 폴더에 저장)
    prepare_all_dataset(json_dir, images_dir, all_dir)
    
    # 2. all 폴더의 데이터를 회전 증강 (augmented 폴더에 10배 생성)
    all_images_dir = os.path.join(all_dir, "images")
    all_labels_dir = os.path.join(all_dir, "labels")
    aug_images_dir = os.path.join(aug_dir, "images")
    aug_labels_dir = os.path.join(aug_dir, "labels")
    os.makedirs(aug_images_dir, exist_ok=True)
    os.makedirs(aug_labels_dir, exist_ok=True)
    augment_all_data(all_images_dir, all_labels_dir, aug_images_dir, aug_labels_dir, num_aug=10)
    print("[INFO] Augmentation done. (원본 10개 → 증강 후 총 약 100개)")
    
    # 3. 증강된 데이터를 train/val로 분할 (예: 80:20)
    train_images_dir, train_labels_dir, val_images_dir, val_labels_dir = split_dataset(aug_images_dir, aug_labels_dir, final_dir, test_size=0.2)
    
    # 4. YOLO 학습용 data.yaml 생성
    data_yaml_path = os.path.join(final_dir, "data.yaml")
    data_yaml_content = f"""train: {os.path.abspath(train_images_dir)}
val: {os.path.abspath(val_images_dir)}
nc: {len(LABEL_MAP)}
names: ["모델명", "사이즈", "DOT"]
"""
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)
    print(f"[INFO] data.yaml created at: {data_yaml_path}")
    
    # 5. YOLOv8 모델 학습 (사전학습 없이 from scratch)
    model = YOLO("yolov8n.yaml")
    model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        name="yolov8n_scratch_with_rotation_aug"
    )
    print("[INFO] Training completed.")
