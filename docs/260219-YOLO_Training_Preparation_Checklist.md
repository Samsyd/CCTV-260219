# 커스텀 YOLO 모델 학습 준비 체크리스트

**작성일:** 2026-02-19
**참여 에이전트:** Barry(Quick Flow Solo Dev), Amelia(Dev), Mary(Analyst)
**목적:** 커스텀 YOLO 학습을 위한 단계별 준비 사항 정리

---

## 전체 체크리스트

```
Phase A: 환경 준비
├── [ ] 1. GPU 확인 & CUDA 설치
├── [ ] 2. Python 패키지 설치
└── [ ] 3. Google Colab 계정 준비 (GPU 없을 경우)

Phase B: 데이터 준비
├── [ ] 4. 현장 이미지 수집 (500장+)
├── [ ] 5. 라벨링 도구 셋업 (Roboflow 추천)
├── [ ] 6. 라벨링 작업 수행
└── [ ] 7. 데이터셋 분할 (train/val/test)

Phase C: 학습 준비
├── [ ] 8. dataset.yaml 작성
├── [ ] 9. train.py 스크립트 준비
└── [ ] 10. 학습 실행 & 모니터링
```

---

## Phase A: 환경 준비

### 1. GPU 확인

```bash
# Windows PowerShell에서 실행
nvidia-smi
```

| 결과 | 의미 | 다음 행동 |
|------|------|----------|
| GPU 정보 출력됨 | NVIDIA GPU 있음 | CUDA 설치 → 로컬 학습 가능 |
| 명령어 없음 | GPU 없거나 Intel/AMD | **Google Colab 사용** (무료 T4 GPU) |

GPU가 있다면 CUDA Toolkit 설치 확인:
```bash
nvcc --version
# 없으면 https://developer.nvidia.com/cuda-downloads 에서 설치
```

### 2. Python 패키지 설치

```bash
# 프로젝트 venv에서 실행
pip install ultralytics    # YOLOv8/v9/v10/v11 통합 패키지
pip install roboflow        # Roboflow 데이터셋 다운로드용
pip install albumentations  # 추가 데이터 증강 (선택)
```

업데이트된 requirements.txt:
```
opencv-python
onvif-zeep
pyyaml
easyocr
ultralytics
numpy
PyQt6
roboflow
```

### 3. Google Colab 준비 (GPU 없을 경우)

- Google 계정으로 [colab.research.google.com](https://colab.research.google.com) 접속
- 런타임 유형 → **T4 GPU** 선택
- Google Drive 마운트해서 데이터셋 업로드

---

## Phase B: 데이터 준비

### 4. 이미지 수집

#### 필요한 촬영 조건

| 조건 | 최소 수량 | 왜 필요한가 |
|------|----------|-----------|
| 맑은 날, 정면 | 100장 | 기본 학습 데이터 |
| 흐린 날 / 그림자 | 100장 | 조명 변화 대응 |
| 야간 / 저조도 | 100장 | 24시간 운영 대비 |
| 비스듬한 각도 | 100장 | 다양한 카메라 위치 대응 |
| 비/눈/안개 | 50장 | 악천후 대응 |
| 가까운 거리 | 50장 | 다양한 차량 위치 |
| 먼 거리 | 50장 | 다양한 차량 위치 |

#### 촬영 시 주의사항

- 컨테이너 BIC 코드가 **화면에서 사람 눈으로 읽을 수 있을 정도로** 보여야 함
- 번호판도 마찬가지 — 사람이 읽을 수 없는 이미지는 AI도 못 읽음
- 다양한 컨테이너 회사(MSCU, TEMU, MAEU 등)를 골고루 포함

#### 자동 수집 스크립트

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

#### 공개 데이터셋 보충

| 데이터셋 | 내용 | 수량 | 용도 |
|---------|------|------|------|
| Roboflow Universe "container code" | 컨테이너 코드 라벨링 | 다양 | BIC 코드 탐지 |
| CCPD (Chinese City Parking Dataset) | 번호판 | 25만+ | 번호판 형식 유사 |
| OpenImages v7 | 차량 번호판 클래스 | 수만 | 범용 번호판 |
| AI Hub 한국 번호판 데이터 | 한국 번호판 | 수만 | 한국 번호판 전용 |

```python
# Roboflow에서 공개 데이터셋 다운로드
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("public").project("container-code-detection")
dataset = project.version(1).download("yolov8")
```

### 5~6. 라벨링

#### Roboflow 사용 절차 (권장)

```
1. roboflow.com 무료 가입
2. New Project → Object Detection 선택
3. 이미지 업로드 (드래그 앤 드롭)
4. 클래스 생성: container_code, license_plate
5. 각 이미지에서 해당 영역을 박스로 지정
6. Export → YOLOv8 포맷 다운로드
```

#### 라벨링 도구 비교

| 도구 | 특징 | 비용 | 추천도 |
|------|------|------|--------|
| **Roboflow** | 웹 기반, 자동 라벨링, 증강, YOLO 직접 내보내기 | 무료 1,000장 | ★★★★★ |
| **Label Studio** | 오픈소스, 자체 서버, ML 백엔드 연동 | 무료 | ★★★★ |
| **CVAT** | 오픈소스, 팀 작업 지원, 자동 라벨링 | 무료 | ★★★★ |
| labelImg | 로컬, 가볍고 심플 | 무료 | ★★★ |

#### 반자동 라벨링 전략

```
[1] 사전학습 YOLO로 1차 자동 라벨링 (정확도 ~60%)
     ↓
[2] Roboflow/Label Studio에서 수동 보정 (정확도 → 95%+)
     ↓
[3] 데이터 증강으로 3~5배 확장
     ↓
[4] Train/Val/Test 분리 (70/20/10)
```

#### 작업 시간 예상

- 숙련자: 시간당 ~100장
- 초보자: 시간당 ~30~50장
- 500장 기준: **약 10~15시간 소요**

### 7. 데이터셋 분할

Roboflow에서 자동 분할 가능. 수동으로 할 경우:

```python
import os, shutil, random

images = os.listdir('dataset/all/images')
random.shuffle(images)

total = len(images)
train_end = int(total * 0.7)
val_end = int(total * 0.9)

splits = {
    'train': images[:train_end],
    'val': images[train_end:val_end],
    'test': images[val_end:]
}

for split, files in splits.items():
    os.makedirs(f'dataset/{split}/images', exist_ok=True)
    os.makedirs(f'dataset/{split}/labels', exist_ok=True)
    for f in files:
        shutil.copy(f'dataset/all/images/{f}', f'dataset/{split}/images/{f}')
        label = f.replace('.jpg', '.txt')
        if os.path.exists(f'dataset/all/labels/{label}'):
            shutil.copy(f'dataset/all/labels/{label}', f'dataset/{split}/labels/{label}')

print(f"Train: {len(splits['train'])}장, Val: {len(splits['val'])}장, Test: {len(splits['test'])}장")
```

#### 디렉토리 구조

```
dataset/
├── dataset.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## Phase C: 학습 준비

### 8. dataset.yaml

```yaml
path: ./dataset
train: train/images
val: val/images
test: test/images

names:
  0: container_code
  1: license_plate
  2: container_body
  3: truck
```

### 9~10. 학습 실행

상세 학습 스크립트는 `260219-Custom_YOLO_Training_Guide.md` 참조.

---

## 즉시 실행 우선순위

```
오늘 할 일:
  1. nvidia-smi 실행해서 GPU 확인
  2. pip install ultralytics roboflow
  3. Roboflow 무료 가입 (roboflow.com)

이번 주 할 일:
  4. 현장 이미지 수집 시작 (하루 100장씩, 5일)
  5. 인터넷에서 컨테이너/번호판 이미지 추가 수집

다음 주 할 일:
  6. Roboflow에서 라벨링 (가장 오래 걸림)
  7. 데이터셋 내보내기 → 학습 시작
```

---

**작성:** BMAD Party Mode (Barry, Amelia, Mary)
**최종 업데이트:** 2026-02-19
