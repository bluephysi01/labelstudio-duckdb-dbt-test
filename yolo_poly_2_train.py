import os
import json
import glob
import shutil
import urllib.parse
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import cv2
import albumentations as A
import numpy as np

#############################################
# 1. 기본 함수들 (JSON 파싱, 이미지 경로 추출, 라벨 변환)
#############################################

# 라벨 맵핑 (라벨명 -> 클래스 ID)
LABEL_MAP = {
    "models": 0,
    "size": 1,
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
    Label Studio의 image 필드에서 실제 이미지 경로 추출
    """
    parsed = urllib.parse.urlparse(image_field)
    qs = urllib.parse.parse_qs(parsed.query)
    if "d" in qs:
        return qs["d"][0]
    else:
        return image_field

def calculate_bbox_from_polygon(points):
    """
    폴리곤 포인트에서 바운딩 박스 계산
    points: [[x1,y1], [x2,y2], ...] 형태의 리스트 (퍼센트 값)
    return: [x_min, y_min, x_max, y_max] (퍼센트 값)
    """
    points_array = np.array(points)
    x_min = np.min(points_array[:, 0])
    y_min = np.min(points_array[:, 1])
    x_max = np.max(points_array[:, 0])
    y_max = np.max(points_array[:, 1]) 
    return [x_min, y_min, x_max, y_max]

def convert_bbox_to_yolo_format(bbox):
    """
    [x_min, y_min, x_max, y_max] 형태의 퍼센트 값 바운딩 박스를
    YOLO 형식 [center_x, center_y, width, height] (0~1 범위)로 변환
    """
    x_min, y_min, x_max, y_max = bbox
    
    # 이미 퍼센트 값이므로 100으로 나눔
    x_min = x_min / 100.0
    y_min = y_min / 100.0
    x_max = x_max / 100.0
    y_max = y_max / 100.0
    
    # YOLO 형식으로 변환 (center_x, center_y, width, height)
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    
    return [center_x, center_y, width, height]

def convert_annotation(json_path, temp_labels_dir, temp_polygon_dir):
    """
    폴리곤 어노테이션 JSON 파일을 YOLO 바운딩 박스 형식으로 변환하고
    원본 폴리곤 좌표도 별도 저장
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
    polygon_file_path = os.path.join(temp_polygon_dir, base_filename + ".json")
    yolo_lines = []
    polygon_data = []
    
    for anno in results:
        value = anno.get("value", {})
        
        # 폴리곤 어노테이션 처리
        if "polygonlabels" in value and "points" in value:
            labels = value.get("polygonlabels", [])
            if not labels:
                continue
                
            label_name = labels[0]
            if label_name not in LABEL_MAP:
                print(f"Warning: label '{label_name}' not in LABEL_MAP. Skipping.")
                continue
                
            class_id = LABEL_MAP[label_name]
            
            # 포인트 추출 (이미 퍼센트 값임)
            points = value.get("points", [])
            if not points:
                continue
            
            # 원본 폴리곤 데이터 저장
            polygon_data.append({
                "class_id": class_id,
                "points": points  # 퍼센트 값 (0-100)
            })
                
            # 바운딩 박스 계산
            bbox = calculate_bbox_from_polygon(points)
            
            # YOLO 형식으로 변환
            yolo_box = convert_bbox_to_yolo_format(bbox)
            
            center_x, center_y, width, height = yolo_box
            line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(line)
    
    if not yolo_lines:
        print(f"Warning: No valid annotations in '{json_path}'. Skipping.")
        return None
    
    # YOLO 형식 라벨 저장
    with open(label_file_path, 'w', encoding='utf-8') as f:
        for line in yolo_lines:
            f.write(line + "\n")
    
    # 원본 폴리곤 좌표 저장
    with open(polygon_file_path, 'w', encoding='utf-8') as f:
        json.dump(polygon_data, f)
        
    return image_file

#############################################
# 2. "all" 데이터셋 준비 (원본 JSON→YOLO 변환)
#############################################
def prepare_all_dataset(json_dir, images_dir, output_all_dir):
    """
    JSON 파일들을 파싱하여 원본 이미지, YOLO 라벨(.txt), 폴리곤 좌표(.json)를
    output_all_dir (예: ./yolo_dataset/all) 에 복사
    """
    images_all_dir = os.path.join(output_all_dir, "images")
    labels_all_dir = os.path.join(output_all_dir, "labels")
    polygons_all_dir = os.path.join(output_all_dir, "polygons")  # 폴리곤 좌표 저장 디렉토리
    os.makedirs(images_all_dir, exist_ok=True)
    os.makedirs(labels_all_dir, exist_ok=True)
    os.makedirs(polygons_all_dir, exist_ok=True)
    
    temp_labels_dir = os.path.join(output_all_dir, "temp_labels")
    temp_polygon_dir = os.path.join(output_all_dir, "temp_polygons")
    os.makedirs(temp_labels_dir, exist_ok=True)
    os.makedirs(temp_polygon_dir, exist_ok=True)
    
    json_files = load_json_files_any_extension(json_dir)
    dataset_entries = []
    for json_file in json_files:
        image_relative_path = convert_annotation(json_file, temp_labels_dir, temp_polygon_dir)
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
        polygon_filename = os.path.splitext(base_filename)[0] + ".json"
        
        src_label_path = os.path.join(temp_labels_dir, label_filename)
        dest_label_path = os.path.join(labels_all_dir, label_filename)
        
        src_polygon_path = os.path.join(temp_polygon_dir, polygon_filename)
        dest_polygon_path = os.path.join(polygons_all_dir, polygon_filename)
        
        if os.path.exists(src_label_path):
            shutil.move(src_label_path, dest_label_path)
        else:
            print(f"Warning: Label file not found: {src_label_path}")
            
        if os.path.exists(src_polygon_path):
            shutil.move(src_polygon_path, dest_polygon_path)
        else:
            print(f"Warning: Polygon file not found: {src_polygon_path}")
    
    shutil.rmtree(temp_labels_dir)
    shutil.rmtree(temp_polygon_dir)
    print(f"[INFO] All dataset prepared at: {output_all_dir}")
    return output_all_dir

#############################################
# 3. 회전 증강 설정 및 관련 함수
#############################################
def load_polygon_data(polygon_path):
    """
    저장된 폴리곤 정보 JSON 파일 로드
    """
    with open(polygon_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def percent_to_absolute(points, img_width, img_height):
    """
    퍼센트 값 (0-100)의 좌표를 절대 좌표로 변환
    """
    abs_points = []
    for x, y in points:
        abs_points.append([x * img_width / 100.0, y * img_height / 100.0])
    return abs_points

def absolute_to_percent(points, img_width, img_height):
    """
    절대 좌표를 퍼센트 값 (0-100)으로 변환
    """
    percent_points = []
    for x, y in points:
        percent_points.append([x * 100.0 / img_width, y * 100.0 / img_height])
    return percent_points

def polygon_to_yolo(polygon, img_width, img_height):
    """
    폴리곤 좌표에서 YOLO 형식의 바운딩 박스 계산
    """
    points_array = np.array(polygon)
    x_min = np.min(points_array[:, 0])
    y_min = np.min(points_array[:, 1])
    x_max = np.max(points_array[:, 0])
    y_max = np.max(points_array[:, 1])
    
    # YOLO 형식으로 변환 (center_x, center_y, width, height)
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    
    # 정규화
    center_x = center_x / img_width
    center_y = center_y / img_height
    width = width / img_width
    height = height / img_height
    
    return [center_x, center_y, width, height]

def write_yolo_label(label_path, boxes, class_ids):
    """
    boxes, class_ids를 YOLO .txt 형식으로 기록
    """
    with open(label_path, 'w', encoding='utf-8') as f:
        for box, cid in zip(boxes, class_ids):
            cx, cy, w, h = box
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def augment_all_data(input_images_dir, input_polygons_dir, output_aug_images_dir, output_aug_labels_dir, num_aug=10):
    """
    input_images_dir, input_polygons_dir에 있는 원본 이미지와 폴리곤 좌표를
    회전 증강하여 output_aug_images_dir, output_aug_labels_dir에 저장.
    
    원본 폴리곤을 회전시키고, 회전된 폴리곤을 포함하는 최소 바운딩 박스 생성
    """
    os.makedirs(output_aug_images_dir, exist_ok=True)
    os.makedirs(output_aug_labels_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_images_dir, "*.*"))
    
    for img_path in image_paths:
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        polygon_path = os.path.join(input_polygons_dir, base_name + ".json")
        
        if not os.path.exists(polygon_path):
            print(f"Warning: Polygon file not found: {polygon_path}")
            continue
        
        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image: {img_path}")
            continue
            
        img_height, img_width = img.shape[:2]
        
        # 폴리곤 데이터 로드
        polygon_data = load_polygon_data(polygon_path)
        if len(polygon_data) == 0:
            print(f"Warning: No polygon data in: {polygon_path}")
            continue
        
        # 각 폴리곤의 모든 포인트를 절대 좌표로 변환하고 키포인트 리스트에 추가
        all_keypoints = []
        class_ids = []
        polygon_points_count = []  # 각 폴리곤의 포인트 개수 기록
        
        for poly in polygon_data:
            points = poly["points"]  # 퍼센트 값 (0-100)
            class_id = poly["class_id"]
            abs_points = percent_to_absolute(points, img_width, img_height)
            
            # 각 폴리곤의 포인트 개수 기록
            polygon_points_count.append(len(abs_points))
            
            # 키포인트 리스트에 절대 좌표 추가
            all_keypoints.extend(abs_points)
            class_ids.append(class_id)
        
        for i in range(num_aug):
            # 회전 변환 설정 (랜덤 각도)
            angle = np.random.uniform(-180, 180)
            keypoint_transform = A.Compose([
                A.Rotate(limit=[angle, angle], p=1.0, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0))
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
            
            # 이미지와 키포인트 함께 회전
            transformed = keypoint_transform(image=img, keypoints=all_keypoints)
            aug_img = transformed['image']
            rotated_keypoints = transformed['keypoints']
            
            # 회전된 키포인트를 다시 폴리곤으로 그룹화
            rotated_polygons = []
            start_idx = 0
            for count in polygon_points_count:
                polygon_points = rotated_keypoints[start_idx:start_idx+count]
                rotated_polygons.append(polygon_points)
                start_idx += count
            
            # 각 회전된 폴리곤에서 YOLO 형식 바운딩 박스 계산
            rotated_boxes = [polygon_to_yolo(poly, img_width, img_height) for poly in rotated_polygons]
            
            # 증강된 이미지 저장
            aug_img_name = f"{base_name}_aug_{i}{ext}"
            aug_img_path = os.path.join(output_aug_images_dir, aug_img_name)
            cv2.imwrite(aug_img_path, aug_img)
            
            # 증강된 라벨 저장
            aug_label_name = f"{base_name}_aug_{i}.txt"
            aug_label_path = os.path.join(output_aug_labels_dir, aug_label_name)
            write_yolo_label(aug_label_path, rotated_boxes, class_ids)

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
    all_polygons_dir = os.path.join(all_dir, "polygons")  # 폴리곤 데이터 디렉토리
    aug_images_dir = os.path.join(aug_dir, "images")
    aug_labels_dir = os.path.join(aug_dir, "labels")
    os.makedirs(aug_images_dir, exist_ok=True)
    os.makedirs(aug_labels_dir, exist_ok=True)
    augment_all_data(all_images_dir, all_polygons_dir, aug_images_dir, aug_labels_dir, num_aug=10)
    print("[INFO] Augmentation done. (원본 10개 → 증강 후 총 약 100개)")
    
    # 3. 증강된 데이터를 train/val로 분할 (예: 80:20)
    train_images_dir, train_labels_dir, val_images_dir, val_labels_dir = split_dataset(aug_images_dir, aug_labels_dir, final_dir, test_size=0.2)
    
    # 4. YOLO 학습용 data.yaml 생성
    data_yaml_path = os.path.join(final_dir, "data.yaml")
    data_yaml_content = f"""train: {os.path.abspath(train_images_dir)}
val: {os.path.abspath(val_images_dir)}
nc: {len(LABEL_MAP)}
names: {list(LABEL_MAP.keys())}
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