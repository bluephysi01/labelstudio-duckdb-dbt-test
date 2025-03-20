import os
import json
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.font_manager as fm
from scipy.ndimage import rotate

def setup_korean_font():
    """한글 폰트 설정"""
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    if '맑은 고딕' not in fm.findSystemFonts(fontpaths=None):
        for font_family in ['NanumGothic', 'AppleGothic', 'Gulim', 'Dotum', 'Batang']:
            if any(font_family.lower() in f.lower() for f in fm.findSystemFonts(fontpaths=None)):
                plt.rcParams['font.family'] = font_family
                break

def read_yolo_label(label_path):
    """YOLO 형식 라벨 파일 읽기"""
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

def read_polygon_file(polygon_path):
    """저장된 폴리곤 JSON 파일 읽기"""
    if not os.path.exists(polygon_path):
        return []
    
    with open(polygon_path, 'r', encoding='utf-8') as f:
        polygon_data = json.load(f)
    return polygon_data

def read_original_json(json_dir, base_name):
    """원본 JSON 파일에서 폴리곤 데이터 읽기"""
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if "task" in data and "data" in data["task"] and "image" in data["task"]["data"]:
                image_path = data["task"]["data"]["image"]
                if base_name in image_path:
                    return data
        except Exception as e:
            print(f"JSON 파일 읽기 오류 {json_file}: {e}")
    
    return None

def extract_polygons_from_json(json_data, img_width, img_height):
    """JSON에서 폴리곤 데이터 추출"""
    polygons = []
    class_names = []
    
    if not json_data or "result" not in json_data:
        return polygons, class_names
    
    for result in json_data["result"]:
        if "value" in result and "points" in result["value"] and "polygonlabels" in result["value"]:
            points = result["value"]["points"]
            labels = result["value"]["polygonlabels"]
            
            if points and labels:
                # 포인트를 픽셀 좌표로 변환 (% -> 픽셀)
                pixel_points = []
                for x, y in points:
                    px = (x / 100.0) * img_width
                    py = (y / 100.0) * img_height
                    pixel_points.append([px, py])
                
                polygons.append(np.array(pixel_points))
                class_names.append(labels[0])
    
    return polygons, class_names

def rotate_point(origin, point, angle_deg):
    """
    한 점을 원점을 기준으로 회전시키는 함수
    """
    ox, oy = origin
    px, py = point
    
    # 각도를 라디안으로 변환
    angle_rad = np.deg2rad(angle_deg)
    
    # 회전 변환 적용
    qx = ox + np.cos(angle_rad) * (px - ox) - np.sin(angle_rad) * (py - oy)
    qy = oy + np.sin(angle_rad) * (px - ox) + np.cos(angle_rad) * (py - oy)
    
    return [qx, qy]

def rotate_polygon(polygon, center, angle_deg):
    """
    폴리곤의 모든 점을 회전시키는 함수
    """
    rotated_points = []
    for point in polygon:
        rotated_point = rotate_point(center, point, angle_deg)
        rotated_points.append(rotated_point)
    
    return np.array(rotated_points)

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

def visualize_original_and_augmented(dataset_dir, json_dir, image_name, save_path=None, rotation_angle=-45):
    """원본 이미지와 증강된 이미지를 나란히 시각화"""
    # 한글 폰트 설정
    setup_korean_font()
    
    # 클래스 이름 목록
    class_names = ["models", "size", "DOT"]
    
    # 색상 정의
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
    
    # 경로 설정
    original_images_dir = os.path.join(dataset_dir, "all", "images")
    original_labels_dir = os.path.join(dataset_dir, "all", "labels")
    original_polygons_dir = os.path.join(dataset_dir, "all", "polygons")
    augmented_images_dir = os.path.join(dataset_dir, "augmented", "images")
    augmented_labels_dir = os.path.join(dataset_dir, "augmented", "labels")
    
    # 이미지 경로 설정
    orig_img_path = os.path.join(original_images_dir, image_name)
    base_name = os.path.splitext(image_name)[0]
    aug_img_path = os.path.join(augmented_images_dir, f"{base_name}_aug_0{os.path.splitext(image_name)[1]}")
    
    orig_label_path = os.path.join(original_labels_dir, f"{base_name}.txt")
    orig_polygon_path = os.path.join(original_polygons_dir, f"{base_name}.json")
    aug_label_path = os.path.join(augmented_labels_dir, f"{base_name}_aug_0.txt")
    
    # 2열 그리드 생성
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # 원본 이미지와 바운딩 박스 표시
    ax1 = axes[0]
    orig_img = cv2.imread(orig_img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    height, width = orig_img.shape[:2]
    ax1.imshow(orig_img)
    
    # 원본 폴리곤 데이터 가져오기
    orig_polygons = []
    orig_poly_classes = []
    
    # 폴리곤 파일에서 먼저 시도
    if os.path.exists(orig_polygon_path):
        polygon_data = read_polygon_file(orig_polygon_path)
        for poly in polygon_data:
            points = poly["points"]
            class_id = poly["class_id"]
            
            # 퍼센트 값을 픽셀 좌표로 변환
            pixel_points = []
            for x, y in points:
                px = (x / 100.0) * width
                py = (y / 100.0) * height
                pixel_points.append([px, py])
            
            orig_polygons.append(np.array(pixel_points))
            if class_id < len(class_names):
                orig_poly_classes.append(class_names[class_id])
            else:
                orig_poly_classes.append(f"Unknown-{class_id}")
    
    # 폴리곤 파일이 없으면 JSON에서 시도
    if not orig_polygons:
        json_data = read_original_json(json_dir, base_name)
        orig_polygons, orig_poly_classes = extract_polygons_from_json(json_data, width, height)
    
    # 폴리곤 데이터 확인
    print(f"원본 폴리곤 수: {len(orig_polygons)}")
    
    # 원본 폴리곤 시각화 (노란색 실선으로 표시)
    if orig_polygons:
        for poly, class_name in zip(orig_polygons, orig_poly_classes):
            # 노란색 실선으로 폴리곤 그리기
            polygon = Polygon(poly, linewidth=1.2, edgecolor='yellow', 
                             facecolor='none', alpha=0.9, linestyle='-')
            ax1.add_patch(polygon)
            
            # 라벨 표시
            try:
                class_idx = class_names.index(class_name)
                color = colors[class_idx]
            except ValueError:
                color = (0.5, 0.5, 0.5, 1.0)
                
            centroid_x = np.mean(poly[:, 0])
            centroid_y = np.mean(poly[:, 1])
            ax1.text(centroid_x, centroid_y, f"{class_name}", 
                    color='white', fontsize=10, 
                    bbox=dict(facecolor=color, alpha=0.8, pad=1))
    
    # 원본 바운딩 박스 표시
    orig_boxes, orig_class_ids = read_yolo_label(orig_label_path)
    
    for box, class_id in zip(orig_boxes, orig_class_ids):
        cx, cy, w, h = box
        class_name = class_names[class_id]
        color = colors[class_id]
        
        # YOLO 좌표를 픽셀 좌표로 변환
        x = int((cx - w/2) * width)
        y = int((cy - h/2) * height)
        w_px = int(w * width)
        h_px = int(h * height)
        
        # 바운딩 박스 그리기
        rect = Rectangle((x, y), w_px, h_px, linewidth=2, 
                        edgecolor=color, facecolor='none', alpha=0.5)
        ax1.add_patch(rect)
    
    ax1.set_title(f"원본: {base_name}", fontsize=12)
    ax1.axis('off')
    
    # 증강된 이미지와 바운딩 박스 표시
    ax2 = axes[1]
    if os.path.exists(aug_img_path) and os.path.exists(aug_label_path):
        aug_img = cv2.imread(aug_img_path)
        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
        aug_height, aug_width = aug_img.shape[:2]
        ax2.imshow(aug_img)
        
        # 회전된 폴리곤 표시
        if orig_polygons:
            # 이미지 중심을 기준으로 회전
            center = (aug_width / 2, aug_height / 2)
            
            # 원본 폴리곤에 동일한 회전 변환 적용
            for poly, class_name in zip(orig_polygons, orig_poly_classes):
                # 폴리곤 회전
                rotated_poly = rotate_polygon(poly, center, rotation_angle)
                
                # 노란색 실선으로 회전된 폴리곤 그리기
                rotated_polygon = Polygon(rotated_poly, linewidth=1.2, edgecolor='yellow', 
                                         facecolor='none', alpha=0.9, linestyle='-')
                ax2.add_patch(rotated_polygon)
        
        # 증강된 바운딩 박스 표시
        aug_boxes, aug_class_ids = read_yolo_label(aug_label_path)
        
        for box_idx, (box, class_id) in enumerate(zip(aug_boxes, aug_class_ids)):
            cx, cy, w, h = box
            class_name = class_names[class_id]
            color = colors[class_id]
            
            # YOLO 좌표를 픽셀 좌표로 변환
            x = int((cx - w/2) * aug_width)
            y = int((cy - h/2) * aug_height)
            w_px = int(w * aug_width)
            h_px = int(h * aug_height)
            
            # 바운딩 박스 그리기
            rect = Rectangle((x, y), w_px, h_px, linewidth=2, 
                           edgecolor=color, facecolor='none', alpha=0.5)
            ax2.add_patch(rect)
            
            # 라벨 텍스트 표시
            ax2.text(x, y-5, f"{class_name}", 
                    color='white', fontsize=10, 
                    bbox=dict(facecolor=color, alpha=0.8, pad=1))
        
        ax2.set_title(f"회전 증강: {os.path.basename(aug_img_path)}", fontsize=12)
    else:
        ax2.text(0.5, 0.5, "증강된 이미지를 찾을 수 없습니다", 
                horizontalalignment='center', verticalalignment='center')
    ax2.axis('off')
    
    plt.tight_layout(pad=2.0)
    plt.suptitle("원본 vs 회전 증강 이미지 비교 (바운딩박스 및 폴리곤)", fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.9)
    
    # 결과 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"비교 시각화 이미지를 {save_path}에 저장했습니다.")
    
    plt.show()
    return fig

def find_optimal_rotation_angle(dataset_dir, image_name):
    """
    증강된 이미지에서 원본 대비 회전 각도 추정
    """
    original_images_dir = os.path.join(dataset_dir, "all", "images")
    augmented_images_dir = os.path.join(dataset_dir, "augmented", "images")
    
    base_name = os.path.splitext(image_name)[0]
    orig_img_path = os.path.join(original_images_dir, image_name)
    aug_img_path = os.path.join(augmented_images_dir, f"{base_name}_aug_0{os.path.splitext(image_name)[1]}")
    
    if not os.path.exists(orig_img_path) or not os.path.exists(aug_img_path):
        print("원본 또는 증강 이미지를 찾을 수 없습니다.")
        return -45  # 기본값 반환
    
    # 이미지 로드
    orig_img = cv2.imread(orig_img_path, cv2.IMREAD_GRAYSCALE)
    aug_img = cv2.imread(aug_img_path, cv2.IMREAD_GRAYSCALE)
    
    # 이미지 크기 맞추기
    if orig_img.shape != aug_img.shape:
        aug_img = cv2.resize(aug_img, (orig_img.shape[1], orig_img.shape[0]))
    
    # 특징점 검출기 생성
    orb = cv2.ORB_create()
    
    # 특징점 및 디스크립터 계산
    kp1, des1 = orb.detectAndCompute(orig_img, None)
    kp2, des2 = orb.detectAndCompute(aug_img, None)
    
    # 특징점 매칭
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        print("충분한 특징점을 찾을 수 없습니다.")
        return -45
    
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # 매칭된 특징점 추출
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
    
    # 변환 행렬 계산
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 회전 각도 추출
    if M is not None:
        # 회전 성분 추출
        angle = np.rad2deg(np.arctan2(M[1, 0], M[0, 0]))
        print(f"추정된 회전 각도: {angle:.2f}도")
        return angle
    
    print("변환 행렬을 계산할 수 없습니다.")
    return -45  # 기본값 반환

# 실행에 필요한 설정값
dataset_dir = './yolo_dataset'
json_dir = './data/export'
image_name = '20250307_123258.jpeg'  # 사용할 이미지 파일명
save_path = 'visualization_result.png'

# 최적 회전 각도 찾기 (옵션)
try:
    rotation_angle = find_optimal_rotation_angle(dataset_dir, image_name)
except Exception as e:
    print(f"회전 각도 자동 추정 실패: {e}")
    rotation_angle = -45  # 기본 회전 각도

print(f"시각화에 사용할 회전 각도: {rotation_angle}도")

# 시각화 함수 호출
visualize_original_and_augmented(dataset_dir, json_dir, image_name, save_path, rotation_angle=rotation_angle)





