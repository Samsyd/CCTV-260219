# 커스텀 YOLO 모델 학습 가이드 — 컨테이너 BIC 코드 & 한국 번호판

**작성일:** 2026-02-19
**참여 에이전트:** Amelia(Dev), Winston(Architect), Barry(Quick Flow Solo Dev), Mary(Analyst)
**목적:** 컨테이너/차량 번호 인식률 향상을 위한 커스텀 YOLO 학습 전체 파이프라인

---

## 1. 개요

현재 시스템(`ai_lpr_test.py`)은 COCO 사전학습 YOLOv8n으로 차량 클래스(`classes=[2,5,6,7]`)만 탐지 후 전체 ROI에서 OCR을 수행한다.
커스텀 모델은 **BIC 코드 영역/번호판 영역 자체를 직접 탐지**하여 OCR 대상을 극적으로 줄이고 정확도와 속도를 모두 향상시킨다.

```
[현재]  프레임 → 차량 탐지 → 차량 전체에 OCR → 정규식 필터
[개선]  프레임 → BIC영역/번호판 직접 탐지 → 해당 영역만 OCR → 형식 검증
```

---

## 2. 커스텀 클래스 정의

```yaml
# dataset.yaml
path: ./dataset
train: train/images
val: val/images
test: test/images

names:
  0: container_code    # 컨테이너 BIC 코드 영역 (영문4+숫자7)
  1: license_plate     # 한국 번호판 영역
  2: container_body    # 컨테이너 본체
  3: truck             # 컨테이너 트럭
```

---

## 3. STEP 1 — 데이터셋 준비

### 3.1 데이터 수집 방법

**방법 1: 현장 카메라 자동 수집**

```python
import cv2, os, time

def auto_capture(save_dir="dataset/raw", interval=2):
    """게이트 카메라에서 자동 이미지 수집"""
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        filename = f"{save_dir}/{time.strftime('%Y%m%d_%H%M%S')}_{count}.jpg"
        cv2.imwrite(filename, frame)
        count += 1
        print(f"캡처: {filename} (총 {count}장)")
        time.sleep(interval)
        if count >= 500:
            break
    cap.release()
```

**방법 2: 공개 데이터셋 활용**

| 데이터셋 | 내용 | 수량 | 용도 |
|---------|------|------|------|
| Roboflow Universe "container code" | 컨테이너 코드 라벨링 | 다양 | BIC 코드 탐지 |
| CCPD (Chinese City Parking Dataset) | 번호판 | 25만+ | 번호판 형식 유사 |
| OpenImages v7 | 차량 번호판 클래스 | 수만 | 범용 번호판 |
| AI Hub 한국 번호판 데이터 | 한국 번호판 | 수만 | 한국 번호판 전용 |

**방법 3: 합성 데이터 생성**

컨테이너 BIC 코드 폰트를 실제 컨테이너 이미지에 합성하고 Albumentations로 다양한 환경 조건을 시뮬레이션한다.

### 3.2 필요한 최소 데이터 수량

| 클래스 | 최소 | 권장 | 고정밀 |
|--------|------|------|--------|
| container_code | 300장 | 1,000장 | 3,000장+ |
| license_plate | 500장 | 2,000장 | 5,000장+ |
| container_body | 200장 | 500장 | 1,500장+ |

### 3.3 디렉토리 구조

```
dataset/
├── dataset.yaml
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── labels/
│       ├── img_001.txt
│       └── img_002.txt
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 3.4 YOLO 라벨 포맷

```
# 각 .txt 파일에 한 줄당 하나의 객체
# class_id  center_x  center_y  width  height  (모두 0~1 정규화)

0 0.523438 0.284722 0.156250 0.068056    ← container_code 영역
1 0.415625 0.720833 0.178125 0.058333    ← license_plate 영역
```

---

## 4. STEP 1.5 — 라벨링 파이프라인

### 4.1 라벨링 도구 비교

| 도구 | 특징 | 비용 | 추천도 |
|------|------|------|--------|
| **Roboflow** | 웹 기반, 자동 라벨링, 증강, YOLO 직접 내보내기 | 무료 1,000장 | ★★★★★ |
| **Label Studio** | 오픈소스, 자체 서버, ML 백엔드 연동 | 무료 | ★★★★ |
| **CVAT** | 오픈소스, 팀 작업 지원, 자동 라벨링 | 무료 | ★★★★ |
| labelImg | 로컬, 가볍고 심플 | 무료 | ★★★ |

### 4.2 반자동 라벨링 전략 (권장)

```
[1] 사전학습 YOLO로 1차 자동 라벨링 (정확도 ~60%)
     ↓
[2] Roboflow/Label Studio에서 수동 보정 (정확도 → 95%+)
     ↓
[3] 데이터 증강으로 3~5배 확장
     ↓
[4] Train/Val/Test 분리 (70/20/10)
```

### 4.3 1차 자동 라벨링 스크립트

```python
import glob
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # 사전학습 모델

for img_path in glob.glob('dataset/raw/*.jpg'):
    results = model(img_path, conf=0.3)
    for r in results:
        boxes = r.boxes
        img_h, img_w = r.orig_shape
        label_path = img_path.replace('.jpg', '.txt')
        with open(label_path, 'w') as f:
            for box in boxes:
                cls = int(box.cls[0])
                x, y, w, h = box.xywhn[0].tolist()
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
```

---

## 5. STEP 2 — 모델 학습

### 5.1 YOLO 버전 선택 (2025~2026 기준)

| 모델 | mAP50 | 속도 (ms) | 파라미터 | 추천 용도 |
|------|-------|----------|---------|----------|
| YOLOv8n | 37.3 | 1.2 | 3.2M | 라즈베리파이/모바일 |
| **YOLOv8s** | 44.9 | 1.9 | 11.2M | **에지 디바이스 (Jetson)** |
| YOLOv8m | 50.2 | 4.0 | 25.9M | PC (GPU 있을 때) |
| **YOLOv9s** | 46.8 | 2.0 | 7.2M | **최신, 효율 최적** |
| YOLOv10s | 46.3 | 2.5 | 7.2M | NMS-free, 후처리 간편 |
| YOLO11s | 47.0 | 2.5 | 9.4M | 가장 최신, Ultralytics 통합 |

**프로젝트 추천: `yolov8s` 또는 `yolo11s`**

### 5.2 학습 전체 스크립트

```python
# train.py — 커스텀 YOLO 학습
from ultralytics import YOLO

# 사전학습 모델 로드 (전이 학습)
model = YOLO('yolov8s.pt')

# 학습 실행
results = model.train(
    data='dataset/dataset.yaml',
    epochs=100,              # 100 에포크 (조기 종료 자동)
    imgsz=640,               # 입력 이미지 크기
    batch=16,                # GPU 메모리에 맞게 조절 (8 or 16)
    patience=20,             # 20 에포크 개선 없으면 조기 종료

    # ── 데이터 증강 (핵심!) ──
    hsv_h=0.015,             # 색상 변환
    hsv_s=0.7,               # 채도 변환
    hsv_v=0.4,               # 밝기 변환
    degrees=5.0,             # 회전 (컨테이너는 약간만)
    translate=0.1,           # 이동
    scale=0.5,               # 스케일 변환
    shear=2.0,               # 전단 변환
    perspective=0.001,       # 원근 변환
    flipud=0.0,              # 상하 반전 (번호판은 하지 않음!)
    fliplr=0.5,              # 좌우 반전
    mosaic=1.0,              # 모자이크 증강
    mixup=0.1,               # 이미지 혼합
    copy_paste=0.1,          # 복사-붙여넣기 증강

    # ── 최적화 ──
    optimizer='AdamW',
    lr0=0.001,               # 초기 학습률
    lrf=0.01,                # 최종 학습률 비율
    warmup_epochs=3,
    cos_lr=True,             # 코사인 학습률 스케줄링

    # ── 출력 ──
    project='runs/train',
    name='container_lpr_v1',
    save=True,
    plots=True,              # 학습 곡선, 혼동 행렬 시각화
)
```

### 5.3 데이터 증강 파라미터 설명

| 파라미터 | 값 | 설명 | 왜 이 값인가 |
|---------|-----|------|------------|
| `degrees` | 5.0 | 회전 각도 | 컨테이너/번호판은 대체로 수평, 과도한 회전 불필요 |
| `flipud` | 0.0 | 상하 반전 비활성 | 번호판/BIC 코드는 뒤집히면 의미 없음 |
| `fliplr` | 0.5 | 좌우 반전 50% | 차량 진입 방향 양쪽 대응 |
| `mosaic` | 1.0 | 모자이크 100% | 작은 객체 탐지에 매우 효과적 |
| `perspective` | 0.001 | 원근 변환 | 카메라 각도 변화 시뮬레이션 |
| `hsv_v` | 0.4 | 밝기 변환 | 주간/야간/그림자 환경 대응 |

### 5.4 GPU 없이 학습 — Google Colab 활용

```python
# Google Colab (무료 T4 GPU)에서 실행

!pip install ultralytics

from google.colab import drive
drive.mount('/content/drive')

# Drive에 업로드한 데이터셋 경로 지정
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.train(
    data='/content/drive/MyDrive/dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# 학습 완료 후 best.pt 다운로드
# runs/train/container_lpr_v1/weights/best.pt
```

**Colab 사용 팁:**
- 무료 계정: T4 GPU, 최대 ~12시간 세션
- Colab Pro: A100 GPU, 더 긴 세션
- 데이터셋은 Google Drive에 zip으로 업로드 후 Colab에서 압축 해제가 빠름

---

## 6. STEP 3 — 학습 결과 검증

### 6.1 테스트셋 평가

```python
model = YOLO('runs/train/container_lpr_v1/weights/best.pt')

# 테스트셋 평가
metrics = model.val(data='dataset/dataset.yaml', split='test')
print(f"mAP50:    {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall:    {metrics.box.mr:.3f}")
```

### 6.2 목표 성능 지표

| 지표 | 최소 목표 | 우수 | 설명 |
|------|----------|------|------|
| mAP50 | 0.85 | 0.93+ | 50% IoU에서의 평균 정밀도 |
| mAP50-95 | 0.60 | 0.75+ | 엄격한 IoU 범위 평균 |
| Precision | 0.85 | 0.95+ | 탐지 중 정답 비율 |
| Recall | 0.80 | 0.90+ | 실제 객체 중 탐지 비율 |

### 6.3 개별 이미지 추론 테스트

```python
results = model('test_image.jpg', conf=0.5, save=True)
for r in results:
    for box in r.boxes:
        cls_name = r.names[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"{cls_name}: {conf:.2f} at [{x1},{y1},{x2},{y2}]")
```

### 6.4 성능 미달 시 개선 전략

| mAP50 범위 | 원인 가능성 | 대응 |
|-----------|-----------|------|
| < 0.50 | 데이터 부족 또는 라벨 오류 | 라벨 검수, 데이터 2배 확보 |
| 0.50~0.70 | 데이터 다양성 부족 | 증강 강화, 다양한 조건 추가 |
| 0.70~0.85 | 모델 용량 부족 | yolov8s → yolov8m 업그레이드 |
| 0.85~0.93 | 미세 조정 필요 | 학습률/에포크 조정, TTA 적용 |

---

## 7. STEP 4 — 기존 코드 통합

### 7.1 ai_lpr_test.py 수정 사항

```python
# ── 변경 1: 모델 로드 (ai_lpr_test.py:43) ──

# 변경 전
yolo_model = YOLO('yolov8n.pt')

# 변경 후
yolo_model = YOLO('models/container_lpr_best.pt')


# ── 변경 2: 탐지 로직 (ai_lpr_test.py:56) ──

# 변경 전
yolo_res = yolo_model(img_small, verbose=False, conf=0.15, classes=[2, 5, 6, 7])

# 변경 후 — 커스텀 클래스이므로 classes 필터 불필요, conf 상향 가능
yolo_res = yolo_model(img_small, verbose=False, conf=0.4)


# ── 변경 3: 클래스별 분기 처리 (search_areas 구성 부분) ──

for r in yolo_res:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 스케일 변환 (320→원본)
        ox1 = int(x1 * w / 320)
        oy1 = int(y1 * h / 240)
        ox2 = int(x2 * w / 320)
        oy2 = int(y2 * h / 240)

        if cls == 0:  # container_code → BIC 코드 영역만 크롭 후 OCR
            search_areas.append(('bic', ox1, oy1, ox2, oy2, conf))
        elif cls == 1:  # license_plate → 번호판 영역만 크롭 후 OCR
            search_areas.append(('plate', ox1, oy1, ox2, oy2, conf))
```

### 7.2 핵심 차이점

```
[기존] 차량 전체 ROI → OCR (노이즈 많음, 느림)
[개선] BIC 코드 영역만 → OCR (정밀, 빠름)
       번호판 영역만 → OCR (정밀, 빠름)
```

---

## 8. STEP 5 — 배포 최적화

### 8.1 모델 경량화 변환

```python
model = YOLO('runs/train/container_lpr_v1/weights/best.pt')

# ONNX (범용, CPU 최적화)
model.export(format='onnx', imgsz=640, half=False, simplify=True)

# TensorRT (NVIDIA GPU/Jetson 전용, 최고 속도)
model.export(format='engine', imgsz=640, half=True)

# OpenVINO (Intel CPU 최적화)
model.export(format='openvino', imgsz=640, half=True)
```

### 8.2 변환된 모델 사용 (코드 변경 없음)

```python
model = YOLO('best.onnx')       # ONNX
model = YOLO('best.engine')     # TensorRT
results = model('image.jpg')    # 동일 API
```

### 8.3 플랫폼별 추천 포맷 및 성능

| 환경 | 포맷 | 예상 FPS (640x640) |
|------|------|------------------|
| Windows PC (CPU only) | ONNX | 15~25 fps |
| Windows PC (NVIDIA GPU) | TensorRT FP16 | 80~150 fps |
| Jetson Orin Nano | TensorRT FP16 | 30~60 fps |
| Raspberry Pi 5 | ONNX + NCNN | 5~10 fps |

---

## 9. 실행 로드맵

| 주차 | 작업 | 산출물 |
|------|------|--------|
| **Week 1** | 데이터 수집 | 현장 500~1000장 + 공개 데이터셋 보충 + 합성 데이터 |
| **Week 2** | 라벨링 | 자동 라벨링 → 수동 보정 → 증강 3~5배 확장 |
| **Week 3** | 학습 & 검증 | Colab에서 YOLOv8s 학습, mAP50 > 0.85 목표 |
| **Week 4** | 통합 & 배포 | ai_lpr_test.py 통합, ONNX 변환, 벤치마크 |

**핵심 원칙:** 데이터 품질이 모델 성능의 80%를 결정한다. 라벨링에 시간을 아끼지 말 것.

---

**작성:** BMAD Party Mode (Amelia, Winston, Barry, Mary)
**최종 업데이트:** 2026-02-19
