# 컨테이너 및 차량 번호 인식 시스템 (LPR & CCR) 개발 계획서

## 1. 프로젝트 개요
해상 운송용 컨테이너를 운반하는 차량의 번호판(LPR)과 컨테이너 식별 번호(CCR, BIC Code)를 실시간 영상에서 자동으로 감지, 추출, 저장하는 지능형 관제 시스템 개발.

## 2. 핵심 기술 아키텍처 (최고 성능 지향)

### A. 객체 탐지 (Object Detection)
*   **모델:** YOLOv8 또는 YOLOv11 (최신 경량 고성능 모델)
*   **역할:** 
    *   1단계: 영상 내 트럭, 컨테이너 영역 탐지
    *   2단계: 번호판(License Plate) 및 컨테이너 번호(BIC Code) 영역 정밀 추출 (ROI)

### B. 문자 인식 (OCR Engine)
*   **엔진:** EasyOCR (Deep Learning 기반) + Tesseract (보조)
*   **특징:** 컨테이너의 다양한 폰트와 오염된 상태에서도 높은 인식률을 확보하기 위한 이미지 전처리(CLAHE, Grayscale 변환) 적용.

### C. 데이터 검증 (Validation)
*   **패턴 매칭:** BIC Code 표준 규격(영문 4자리 + 숫자 7자리) 정규표현식 검증.
*   **체크섬 검사:** BIC Code의 마지막 체크 디지트 계산을 통한 데이터 신뢰도 확보.

## 3. 주요 기능 로드맵

### Phase 1: 환경 구축 및 엔진 개발
*   AI 라이브러리(PyTorch, EasyOCR, OpenCV) 설치.
*   이미지 내 번호판/컨테이너 번호 추출 프로토타입 개발.

### Phase 2: 실시간 스트리밍 통합
*   RTSP 영상 스트림 연결 및 프레임별 AI 추론 적용.
*   멀티스레딩을 통한 영상 끊김 방지 및 처리 속도 최적화.

### Phase 3: GUI 및 데이터 관리
*   L자형 레이아웃에 실시간 인식 결과 보드 추가.
*   감지 시 스냅샷 저장 및 CSV/DB 로그 기록.

## 4. 기대 효과
*   물류 게이트 통과 차량의 수동 기록 업무 자동화.
*   데이터 정확도 향상 및 실시간 관제 효율 극대화.

---

## 5. 현재 환경 검토 결과 (2026-02-17)

### A. 하드웨어 현황

| 항목 | 현재 상태 | 비고 |
|------|-----------|------|
| OS | Windows 11 Pro | |
| CPU | Intel i7-11700 (8코어/16스레드, 2.5GHz) | 충분 |
| RAM | 32GB (가용 ~10GB) | 충분 |
| GPU | Intel UHD Graphics 750 (내장 그래픽) | NVIDIA GPU 없음 |

### B. 소프트웨어 현황

| 패키지 | 상태 | 비고 |
|--------|------|------|
| Python | 3.13.9 | PyTorch 호환 위험 (3.11/3.12 권장) |
| OpenCV | 4.13.0 | ✅ 설치 완료 |
| PyQt6 | 6.10.2 | ✅ 설치 완료 |
| ONVIF-zeep | 0.2.12 | ✅ 설치 완료 |
| PyTorch | **미설치** | Phase 1 필수 |
| Ultralytics (YOLO) | **미설치** | Phase 1 필수 |
| EasyOCR | **미설치** | Phase 1 필수 |
| Tesseract OCR | **미설치** | Phase 1 보조 |

### C. 기존 구현 완료 항목
*   ✅ 네트워크 카메라 스캐너 (`cctv_scanner.py`)
*   ✅ ONVIF 장치 탐색 (`cctv_discovery.py`)
*   ✅ RTSP 스트림 접속 테스트 (`cctv_blind_test.py`)
*   ✅ 카메라 브랜드 감지 (`cctv_brand_info.py`)
*   ✅ PyQt6 L자형 레이아웃 GUI (`cctv_advanced_layout.py`)
*   ✅ 멀티 레이아웃 뷰어 2x2/3x3/4x4 (`cctv_multi_layout.py`)

### D. 미해결 이슈
*   RTSP 카메라 인증 실패 (`401 Unauthorized`) — 카메라 관리자 ID/PW 확보 필요

---

## 6. 추론 모드 선택

> **아래 두 가지 모드 중 하나를 선택하여 구현을 진행합니다.**

---

### 🅰 모드 A: CPU 추론 (현재 환경 그대로 진행)

**추가 하드웨어: 불필요**

| 항목 | 내용 |
|------|------|
| 추론 엔진 | PyTorch CPU + OpenVINO (Intel 최적화) |
| YOLO 모델 | YOLOv8**n** (nano) — 최경량 모델 |
| 예상 성능 | 프레임당 ~50-100ms → **약 3-5 FPS** |
| OCR 속도 | ROI당 ~200-500ms |
| 전체 파이프라인 | **약 1-3 FPS (준실시간)** |

**최적화 전략:**
*   모션 감지 트리거 — 차량 진입 시에만 AI 추론 활성화
*   프레임 스킵 — 매 5~10 프레임마다 추론 수행
*   ONNX Runtime + OpenVINO — Intel CPU 전용 가속
*   경량 모델 사용 — YOLOv8n 고정

**설치 요구사항:**
```
# Python 3.11 또는 3.12로 가상환경 재생성 필요
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics easyocr openvino
```

**추가 디스크:** ~900MB

**적합한 경우:** 게이트 통과 차량이 저속이고, 약간의 인식 지연 허용 가능 시

---

### 🅱 모드 B: GPU 추론 (NVIDIA GPU 추가 장착)

**추가 하드웨어: NVIDIA GPU 필요**

| 항목 | 내용 |
|------|------|
| 권장 GPU | RTX 3060 (12GB) 이상 |
| 추론 엔진 | PyTorch CUDA + TensorRT |
| YOLO 모델 | YOLOv8s/m — 고정밀 모델 사용 가능 |
| 예상 성능 | 프레임당 ~5-10ms → **약 25-30+ FPS** |
| OCR 속도 | ROI당 ~30-50ms |
| 전체 파이프라인 | **약 15-25 FPS (완전 실시간)** |

**이점:**
*   문서 계획서의 "실시간" 목표 완전 달성
*   고정밀 모델(YOLOv8m) 사용으로 인식률 향상
*   다채널 동시 추론 가능 (2-4개 카메라 병렬 처리)
*   향후 확장성 확보

**설치 요구사항:**
```
# Python 3.11 또는 3.12로 가상환경 재생성 필요
# CUDA Toolkit 12.x + cuDNN 설치 선행
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics easyocr
```

**추가 디스크:** ~3.5GB (PyTorch GPU 포함)

**적합한 경우:** 끊김 없는 실시간 인식이 필수이고, 다채널 처리가 요구될 시

---

### 모드 비교 요약

| 비교 항목 | 🅰 CPU 모드 | 🅱 GPU 모드 |
|-----------|------------|------------|
| 추가 비용 | 없음 | GPU 구매 (~30-50만원) |
| 실시간 성능 | 1-3 FPS (준실시간) | 15-25 FPS (완전 실시간) |
| 인식 정밀도 | 보통 (경량 모델) | 높음 (고정밀 모델) |
| 다채널 처리 | 1채널 권장 | 2-4채널 동시 가능 |
| 설치 난이도 | 낮음 | 중간 (CUDA 설정 필요) |
| 확장성 | 제한적 | 우수 |

---

## 7. 공통 선행 작업 (모드 무관)

1.  **Python 가상환경 재생성** — Python 3.11 또는 3.12 사용
2.  **Tesseract OCR 설치** — Windows 바이너리 설치
3.  **RTSP 카메라 자격 증명 확보** — 카메라 관리자에게 ID/PW 요청
4.  **ONVIF 서비스 활성화 확인** — 각 카메라 웹 설정 페이지에서 확인

---

## 8. Phase별 구현 가능 판정

| Phase | 🅰 CPU 모드 | 🅱 GPU 모드 |
|-------|------------|------------|
| Phase 1: 엔진 개발 | ⚠️ 가능 (경량 모델 한정) | ✅ 완전 가능 |
| Phase 2: 실시간 통합 | ⚠️ 가능 (1-3 FPS) | ✅ 완전 가능 (25+ FPS) |
| Phase 3: GUI + 로깅 | ✅ 즉시 가능 | ✅ 즉시 가능 |

---
**작성자:** 노바 (Nova)
**작성일:** 2026-02-17
**검토일:** 2026-02-17 (환경 검토 및 모드 선택 항목 추가)
