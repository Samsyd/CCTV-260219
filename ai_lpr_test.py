import cv2
import easyocr
import time
import os
import re
from collections import Counter
import csv
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import threading

# 전역 변수
latest_frame = None
confirmed_ccr = ""
is_running = True
display_boxes = []

def save_log(log_type, text, prob):
    log_dir = 'logs'
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    filename = f"{log_dir}/{datetime.now().strftime('%Y-%m-%d')}_gate_log.csv"
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([datetime.now().strftime("%H:%M:%S"), log_type, text, f"{prob:.2f}"])

def check_bic_code(text):
    """컨테이너 BIC Code 정밀 검증 (영문4+숫자6~7)"""
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    # 1. 표준 규격 (영문4 + 숫자7)
    match = re.search(r'[A-Z]{4}\d{7}', clean)
    if match: return match.group()
    # 2. 완화 규격 (영문4 + 숫자6)
    match = re.search(r'[A-Z]{4}\d{6}', clean)
    if match: return match.group()
    return None

def ai_worker():
    global latest_frame, confirmed_ccr, display_boxes
    print("Nova Optimized CCR Engine 가동...")
    
    # OCR 엔진 초기화 (가장 강력한 감지 모드)
    reader = easyocr.Reader(['en'], gpu=False)
    yolo_model = YOLO('yolov8n.pt')
    ccr_history = []

    while is_running:
        if latest_frame is None:
            time.sleep(0.1)
            continue

        frame_to_proc = latest_frame.copy()
        h, w = frame_to_proc.shape[:2]
        
        # [1단계] 고감도 물체 탐지 (YOLO)
        img_small = cv2.resize(frame_to_proc, (320, 240))
        yolo_res = yolo_model(img_small, verbose=False, conf=0.15, classes=[2, 5, 6, 7])
        
        search_areas = []
        for r in yolo_res:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 영역 대폭 확장 (컨테이너 박스 외곽 번호 대비)
                margin = 50
                search_areas.append((max(0, int(x1*w/320)-margin), max(0, int(y1*h/240)-margin), 
                                     min(w, int(x2*w/320)+margin), min(h, int(y2*h/240)+margin)))
        
        if not search_areas: search_areas = [(0, 0, w, h)]

        new_boxes = []
        for (sx1, sy1, sx2, sy2) in search_areas:
            roi = frame_to_proc[sy1:sy2, sx1:sx2]
            if roi.size == 0: continue
            
            # [2단계] 저화질/세로형 특화 전처리 (핵심!)
            # 이미지 2배 확대 + 샤프닝 + 적응형 이진화
            roi_zoom = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(roi_zoom, cv2.COLOR_BGR2GRAY)
            
            # 커널 샤프닝으로 글자 경계 복원
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            gray = cv2.filter2D(gray, -1, kernel)
            
            # 적응형 이진화 (그림자/반사 무력화)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)

            # [3단계] 전방위 OCR 분석
            res = reader.readtext(processed, rotation_info=[90, 270], paragraph=True, 
                                 low_text=0.2, link_threshold=0.1, mag_ratio=2.5)
            
            # 낱글자 병합을 위한 리스트
            captured_texts = []
            for (bbox, text) in res:
                clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(clean) > 0:
                    captured_texts.append(clean)
                    # 표시용 박스 계산 (2배 확대 반영)
                    (tl, tr, br, bl) = bbox
                    new_boxes.append(((int(tl[0]/2 + sx1), int(tl[1]/2 + sy1)), 
                                     (int(br[0]/2 + sx1), int(br[1]/2 + sy1)), clean))

            # [핵심] 공간적/논리적 병합 시도
            full_raw = "".join(captured_texts)
            code = check_bic_code(full_raw)
            
            if not code: # 개별 텍스트에서도 한 번 더 확인
                for t in captured_texts:
                    if check_bic_code(t): code = check_bic_code(t); break

            if code:
                ccr_history.append(code)
                print(f"★ 최적화 엔진 포착: {code}")

        display_boxes = new_boxes

        # 투표로 확정 (2회 일치 시)
        if len(ccr_history) > 12: ccr_history.pop(0)
        if ccr_history:
            most = Counter(ccr_history).most_common(1)[0]
            if most[1] >= 2:
                if confirmed_ccr != most[0]:
                    confirmed_ccr = most[0]
                    save_log('CCR', confirmed_ccr, 0.9)

        time.sleep(0.5) # CPU 효율 관리

def main():
    global latest_frame, is_running
    threading.Thread(target=ai_worker, daemon=True).start()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)

    print("컨테이너 최적화 인식 모드 가동! 'q' 종료")

    while True:
        ret, frame = cap.read()
        if not ret: break
        latest_frame = frame.copy()
        h, w = frame.shape[:2]

        # UI 가이드 및 결과
        if confirmed_ccr:
            cv2.rectangle(frame, (10, h-80), (550, h-10), (0, 0, 0), -1)
            cv2.putText(frame, f"FINAL CCR: {confirmed_ccr}", (20, h-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        for (pt1, pt2, txt) in display_boxes:
            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            cv2.putText(frame, txt, (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Nova CCR Optimized Master", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
