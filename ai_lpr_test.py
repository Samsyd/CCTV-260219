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

def save_log(log_type, text, prob):
    log_dir = 'logs'
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"{log_dir}/{today}_gate_log.csv"
    timestamp = datetime.now().strftime("%H:%M:%S")
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Time', 'Type', 'Result', 'Confidence'])
        writer.writerow([timestamp, log_type, text, f"{prob:.2f}"])

def check_bic_code(text):
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    # 실전용: 영문4자 + 숫자 6~7자
    match = re.search(r'[A-Z]{3,4}\d{5,7}', clean)
    return match.group() if match else None

def main():
    if not os.path.exists('captures'): os.makedirs('captures')

    print("Nova Gate-Master Pro (Low-Res Optimized) 초기화 중...")
    yolo_model = YOLO('yolov8n.pt')
    # 감지 민감도를 극대화한 설정
    reader = easyocr.Reader(['en'], gpu=False)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    
    last_ocr_time = time.time()
    ccr_history = []
    ccr_confirmed = ""

    print("시스템 가동: 저화질 현장 모드 (Dilation + High-Sensitivity)")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        
        # [1단계] 물체 감지 생략 및 화면 전체 감시 (저화질에서는 YOLO가 더 방해될 수 있음)
        # 하지만 큰 틀을 잡기 위해 YOLO 결과가 있으면 그 영역을 우선적으로 봅니다.
        img_small = cv2.resize(frame, (320, 240))
        yolo_results = yolo_model(img_small, verbose=False, conf=0.15, classes=[2, 5, 6, 7])
        
        search_areas = []
        scale_x, scale_y = w/320, h/240
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                search_areas.append((int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)))
        
        if not search_areas: search_areas = [(0, 0, w, h)] # 차 안 잡히면 전체 감시

        current_time = time.time()
        if current_time - last_ocr_time > 0.8:
            for (sx1, sy1, sx2, sy2) in search_areas:
                # ROI 추출 및 확장
                roi = frame[max(0, sy1-20):min(h, sy2+20), max(0, sx1-20):min(w, sx2+20)]
                if roi.size == 0: continue
                
                # --- [저화질 전용 전처리 핵심] ---
                # 1. 2배 확대 (Cubic)
                roi_zoom = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(roi_zoom, cv2.COLOR_BGR2GRAY)
                
                # 2. 노이즈 제거 및 대비 강화
                gray = cv2.GaussianBlur(gray, (3,3), 0)
                gray = cv2.equalizeHist(gray)
                
                # 3. 모폴로지 팽창 (끊긴 글자 이어주기)
                kernel = np.ones((2,2), np.uint8)
                dilated = cv2.dilate(gray, kernel, iterations=1)
                
                # 4. 다중 이진화 시도 (Adaptive + Simple)
                processed = cv2.adaptiveThreshold(dilated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)

                # [고민감도 OCR]
                # low_text: 아주 낮은 선명도의 텍스트도 감지 (기본 0.4 -> 0.2)
                # link_threshold: 글자 사이 간격이 멀어도 이어줌 (기본 0.4 -> 0.1)
                res = reader.readtext(processed, rotation_info=[90, 270], paragraph=True, 
                                     low_text=0.2, link_threshold=0.1, mag_ratio=2.5)
                
                raw_full_text = "".join([re.sub(r'[^A-Z0-9]', '', r[1].upper()) for r in res])
                
                # 개별 및 병합 텍스트에서 BIC Code 추출
                code = check_bic_code(raw_full_text)
                if code:
                    ccr_history.append(code)
                    print(f"★ 저화질 엔진 포착: {code}")

            if len(ccr_history) > 15: ccr_history.pop(0)
            if ccr_history:
                counts = Counter(ccr_history)
                most = counts.most_common(1)[0]
                if most[1] >= 2:
                    if ccr_confirmed != most[0]:
                        ccr_confirmed = most[0]
                        save_log('CCR', ccr_confirmed, 0.75)
                        cv2.imwrite(f"captures/CCR_LOWRES_{datetime.now().strftime('%H%M%S')}.jpg", frame)

            last_ocr_time = current_time

        # UI 출력
        if ccr_confirmed:
            cv2.rectangle(frame, (10, h-70), (600, h-10), (0, 0, 0), -1)
            cv2.putText(frame, f"CONFIRMED CCR (LOW-RES): {ccr_confirmed}", (20, h-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Nova Low-Res Master", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
