# BMAD Party Mode 코드 리뷰 & 인식률 최적화 논의 보고서

**작성일:** 2026-02-19
**참여 에이전트:** Winston(Architect), Amelia(Dev), Quinn(QA), Mary(Analyst), Barry(Quick Flow Solo Dev)
**주제:** 코드 품질 리뷰 + 컨테이너/차량 번호 인식률 근본적 개선 전략

---

## 1. 코드 리뷰 — 문제점 요약

### 1.1 아키텍처 문제 (Winston)

| 항목 | 현재 상태 | 심각도 |
|------|----------|--------|
| 모듈화 부재 | 10개 `.py` 파일이 루트에 평탄 나열, 패키지 구조 없음 | High |
| 전역 가변 상태 | `ai_lpr_test.py:14-17` — Lock 없이 스레드 공유 (레이스 컨디션) | Critical |
| 중복 로직 | 이미지 전처리가 `ai_lpr_test.py`와 `ai_sample_tester.py`에 중복 | Medium |
| 설정 관리 부재 | IP, 포트, 자격증명이 각 파일에 하드코딩 | High |

**제안 패키지 구조:**
```
project/
├── config/          # 설정 파일 (IP, 포트, 자격증명)
├── core/            # OCR 엔진, BIC 검증, 전처리 파이프라인
├── network/         # 스캐너, 디스커버리, 브랜드 감지
├── ui/              # PyQt6 뷰어들
└── tests/           # 유닛/통합 테스트
```

### 1.2 코드 품질 문제 (Amelia)

| 파일 | 라인 | 문제 | 심각도 |
|------|------|------|--------|
| `cctv_scanner.py` | :12 | `except: pass` — 모든 예외 무시 | Critical |
| `cctv_brand_info.py` | :25 | Bare `except:` — 에러 정보 소실 | Critical |
| `cctv_multi_viewer.py` | :55 | 루프 내부 `import numpy` | High |
| `ai_lpr_test.py` | :14-17 | Lock 없는 전역 변수 스레드 공유 | High |
| `cctv_advanced_layout.py` | :17 | `VideoThread`에 종료 조건 없음 — 리소스 누수 | High |
| `ai_lpr_test.py` | :131 | `CAP_DSHOW` — Windows 전용, 크로스플랫폼 미지원 | Medium |
| `requirements.txt` | 전체 | `easyocr`, `ultralytics`, `PyQt6`, `numpy` 누락 | Critical |
| `cctv_blind_test.py` | :18-30 | 자격증명 하드코딩 (git에 노출) | Critical |

### 1.3 테스트 커버리지 (Quinn)

**현재 테스트 커버리지: 0%**

즉시 필요한 테스트:
1. `check_bic_code()` 유닛 테스트 — 핵심 비즈니스 로직
2. `extract_bic_parts()` 유닛 테스트 — 조각 병합 로직
3. 스레드 안전성 테스트 — `ai_worker()` 동시 접근
4. `save_log()` 테스트 — CSV 생성/추가 로직
5. `requirements.txt` 불일치 해결 (`pip freeze > requirements.txt`)

### 1.4 비즈니스 로직 분석 (Mary)

- BIC 코드 검증 전략이 파일별로 불일치 (투표 기반 vs 조각 병합)
- 확정 임계값(`>=2`)이 너무 낮아 오탐 위험
- YOLO `classes=[2,5,6,7]`은 컨테이너 자체를 탐지하지 못함 (차량에 실린 것만 간접 탐지)
- 로그 데이터 수집만 하고 분석/활용 기능 없음

---

## 2. 인식률 근본적 개선 전략

### 2.1 LAYER 1: 하드웨어 & 장비

#### 현재 입력의 근본적 한계

`ai_lpr_test.py:130`에서 `cv2.VideoCapture(0)` — 일반 웹캠 사용.
아무리 AI가 뛰어나도 입력 데이터 품질을 넘을 수 없다.

#### 웹캠 vs 산업용 카메라 비교

| 항목 | 일반 웹캠 (현재) | 산업용 카메라 (필요) |
|------|-----------------|-------------------|
| 해상도 | 720p~1080p | 2MP~5MP |
| 셔터 | 롤링 셔터 | 글로벌 셔터 (이동체 왜곡 방지) |
| 렌즈 | 고정 광각 | 가변 초점 / 줌 렌즈 |
| IR | 없음 | IR 적외선 LED + IR 필터 (야간) |
| 동적범위 | 좁음 | WDR/HDR (역광 대응) |

#### 하드웨어 아키텍처 제안 — 3-티어

```
[Tier 1: 캡처]              [Tier 2: 처리]          [Tier 3: 서비스]
 광각 카메라 (진입 감지) ──→  에지 디바이스      ──→  서버/클라우드
 줌 카메라 (번호 정밀촬영) →  (Jetson/PC)           (로그, 대시보드)
 보조 IR 조명
```

#### 현실적 장비 추천

| 용도 | 추천 장비 | 가격대 |
|------|----------|--------|
| 번호판 전용 | Hikvision DS-2CD7A26G0/P-IZS (LPR 특화) | ₩50~80만 |
| 범용 고해상도 | Dahua IPC-HFW5442T-ASE (4MP AI) | ₩30~50만 |
| 에지 처리 | NVIDIA Jetson Orin Nano | ₩25~40만 |
| 저비용 옵션 | Raspberry Pi 5 + HQ Camera Module | ₩15~20만 |

### 2.2 LAYER 2: 소프트웨어 전처리 파이프라인

#### 약점 1: 단일 전처리 경로

현재 `ai_lpr_test.py:76-84`는 하나의 전처리만 적용. 환경 조건이 매 프레임 다르므로 적응형 다중 전처리 필요:

```python
def adaptive_preprocess(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    results = []

    if brightness < 80:       # 어두운 이미지
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        results.append(clahe.apply(gray))
    elif brightness > 180:    # 과노출 이미지
        results.append(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX))
    if contrast < 30:         # 저대비 이미지
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        results.append(clahe.apply(gray))

    results.append(gray)      # 항상 표준 처리 포함
    return results             # 각각에 OCR 시도
```

#### 약점 2: AI 기반 Super-Resolution 미적용

단순 INTER_CUBIC 보간 대신 Real-ESRGAN으로 실제 디테일 복원:

```python
from realesrgan import RealESRGANer
upsampler = RealESRGANer(scale=4, model_path='RealESRGAN_x4plus.pth', ...)
sr_output, _ = upsampler.enhance(roi, outscale=4)
```

#### 약점 3: 원근 보정 없음

컨테이너 번호는 대부분 비스듬한 각도에서 촬영됨. 텍스트 영역 감지 후 4점 원근 변환으로 직사각형 정규화 필요.

#### 추가 개선: 업스케일링 보간법

`INTER_CUBIC` → `INTER_LANCZOS4` 변경 (텍스트 엣지 보존 우수).

#### 추가 개선: ISO 6346 체크디짓 검증

정규식만으로 검증하지 말고 수학적 체크디짓으로 오탐 필터링:

```python
def validate_iso6346(code):
    if len(code) != 11: return False
    values = {chr(i): v for i, v in zip(range(65, 91),
              [10,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,
               28,29,30,31,32,34,35,36,37,38])}
    total = sum((values.get(c, int(c)) if c.isalpha() else int(c)) * (2**i)
               for i, c in enumerate(code[:10]))
    return (total % 11 % 10) == int(code[10])
```

#### 추가 개선: 신뢰도 가중 투표 시스템

현재 `most[1] >= 2` (2회 일치 확정)는 오탐 위험이 높음. 개선안:

```python
from collections import defaultdict
ccr_weighted = defaultdict(float)

if code:
    ccr_weighted[code] += confidence
    for k in list(ccr_weighted):
        ccr_weighted[k] *= 0.95   # 시간 감쇠

best = max(ccr_weighted, key=ccr_weighted.get)
if ccr_weighted[best] >= 3.0:    # 가중 점수 3.0 이상
    confirmed_ccr = best
```

### 2.3 LAYER 3: AI 모델 & 최신 기술 (2025~2026)

#### 텍스트 탐지 (Text Detection)

| 기술 | 정확도 | 속도 | 특징 |
|------|--------|------|------|
| CRAFT (현재 EasyOCR 내장) | ★★★☆ | 느림 | 문자 단위 탐지 |
| DBNet++ | ★★★★ | 빠름 | 실시간 가능, 경량 |
| **PaddleOCR v4 Det** | ★★★★★ | 빠름 | **현시점 최강**, PP-OCRv4 |
| YOLO-World + Text | ★★★☆ | 매우 빠름 | 제로샷 텍스트 영역 탐지 |

#### 텍스트 인식 (Text Recognition)

| 기술 | 정확도 | 속도 | 특징 |
|------|--------|------|------|
| EasyOCR (현재) | ★★★☆ | 보통 | 범용, 설치 간편 |
| **PaddleOCR v4 Rec** | ★★★★★ | 빠름 | **SVTRv2 기반, 현시점 최강** |
| TrOCR (Microsoft) | ★★★★ | 느림 | Transformer 기반 |
| GOT-OCR 2.0 | ★★★★★ | 느림 | 2024 SOTA, 범용 OCR |
| Surya OCR | ★★★★ | 보통 | 다국어 특화 |

#### 번호판 전용 (LPR/ANPR)

| 기술 | 한국 번호판 | 속도 | 특징 |
|------|-----------|------|------|
| OpenALPR | 미지원 | 빠름 | 북미/유럽 특화 |
| Plate Recognizer API | ★★★★★ | 클라우드 | 한국 번호판 지원, 유료 |
| **YOLO + 커스텀 학습** | ★★★★★ | 빠름 | **한국 번호판 데이터셋 파인튜닝** |
| PaddleOCR + 한국 모델 | ★★★★ | 빠름 | 커스텀 학습 용이 |

#### 컨테이너 코드 전용

| 기술 | 정확도 | 방식 |
|------|--------|------|
| ISO 6346 전용 파인튜닝 | ★★★★★ | YOLO 코드 영역 탐지 → OCR |
| Roboflow 컨테이너 모델 | ★★★★ | 사전학습 모델 활용 |
| 커스텀 CRNN | ★★★★ | 시퀀스 인식, 고정 형식 최적 |

#### 추천 최적 스택

```
[1단] YOLOv8s/v9 (차량+컨테이너 탐지, 커스텀 학습)
  ↓
[2단] PaddleOCR v4 텍스트 탐지 (DBNet++ 기반)
  ↓
[3단-A] PaddleOCR v4 인식 (컨테이너 BIC 코드)
[3단-B] YOLO + PaddleOCR 한국 번호판 모델 (차량 번호)
  ↓
[4단] ISO 6346 체크디짓 + 한국 번호판 형식 검증
  ↓
[5단] 신뢰도 가중 투표 시스템
```

---

## 3. 단계별 투자 대비 효과 분석

```
인식률
95% ─────────────────────── ★ Phase 3 (커스텀 학습 + 전용 카메라)
90% ─────────────────╱────
80% ───────────╱────────── ★ Phase 2 (PaddleOCR + 전처리 강화)
65% ─────╱────────────────
45% ╱───────────────────── ★ Phase 1 (현재 + 소프트웨어 최적화)
30% ──────────────────────  현재 추정 인식률
    └──┬──────┬──────┬──→ 투자
      ₩0    ₩30만  ₩100만+
```

### Phase 1 — 즉시 적용 (₩0)
- PaddleOCR 교체 + CLAHE + 다중 전처리 → 예상 +15~20%p
- 신뢰도 가중 투표 시스템 개선 → 예상 +5%p
- ISO 6346 체크디짓 검증 추가

### Phase 2 — 단기 (₩15~30만, 1~2주)
- Raspberry Pi 5 + HQ 카메라 모듈 또는 산업용 USB 카메라
- Real-ESRGAN 초해상도 적용
- 원근 보정 추가 → 예상 +15%p

### Phase 3 — 중기 (₩50~100만+, 1~2개월)
- 전용 LPR 카메라 또는 Jetson + 산업용 카메라
- 한국 번호판/컨테이너 코드 커스텀 YOLO 모델 학습
- ISO 6346 검증 통합 → 예상 +10~15%p

---

## 4. 장기 아키텍처: 엣지-클라우드 하이브리드

```
[현장]                          [클라우드/서버]
카메라 → Jetson/PC
  ↓
실시간 인식 (PaddleOCR)    ──→   인식 실패 이미지 수집
  ↓                             ↓
게이트 제어/로그           ←──   모델 재학습 → 배포
```

인식 실패 이미지를 자동 수집하면 시간이 갈수록 정확도가 올라가는 시스템 구축 가능.

---

## 5. 벤치마크 시스템 (필수 선행 과제)

인식률 개선을 측정하려면 벤치마크가 필수:

1. **Ground Truth 데이터셋** 구축 (container_clear, container_blur, plate_day, plate_night 등)
2. **인식률 측정 스크립트** (정확 일치, 부분 일치, 미인식 비율)
3. **A/B 테스트 프레임워크** (전처리 변경 전후 정량 비교)
4. **파이프라인 단계별 성공률** 독립 측정 (탐지율 × 텍스트영역율 × OCR율 = 전체율)

---

**결론:** Phase 1(소프트웨어 최적화)부터 시작하여 벤치마크 결과를 근거로 Phase 2, 3 투자를 결정하는 것을 권장.

**작성:** BMAD Party Mode (Winston, Amelia, Quinn, Mary, Barry)
**최종 업데이트:** 2026-02-19
