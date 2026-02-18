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

# 성공한 실제 카메라 주소
RTSP_URL = "rtsp://admin:4321@192.168.0.20:554/profile2/media.smp"

def save_log(log_type, text, prob):
    log_dir = 'logs'
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    filename = f"{log_dir}/{datetime.now().strftime('%Y-%m-%d')}_gate_log.csv"
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([datetime.now().strftime("%H:%M:%S"), log_type, text, f"{prob:.2f}"])

def check_bic_code(text):
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    # BIC Code 표준 규격: 영문 4자 + 숫자 6~7자
    match = re.search(r'[A-Z]{4}\d{6,7}', clean)
    return match.group() if match else None

def ai_worker():
    global latest_frame, confirmed_ccr, display_boxes
    print("AI 워커: 최종 정밀 모드 가동 중...")
    
    # 폰트 인식을 위한 최적 설정
    reader = easyocr.Reader(['en'], gpu=False)
    yolo_model = YOLO('yolov8n.pt')
    ccr_history = []

    while is_running:
        if latest_frame is None:
            time.sleep(0.1)
            continue

        frame_to_proc = latest_frame.copy()
        h, w = frame_to_proc.shape[:2]
        
        # [1단계] 차량 및 컨테이너 감지 영역 획득
        img_small = cv2.resize(frame_to_proc, (320, 240))
        yolo_res = yolo_model(img_small, verbose=False, conf=0.15, classes=[2, 5, 6, 7])
        
        search_areas = []
        for r in yolo_res:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 검색 영역을 충분히 확보
                margin = 40
                search_areas.append((max(0, int(x1*w/320)-margin), max(0, int(y1*h/240)-margin), 
                                     min(w, int(x2*w/320)+margin), min(h, int(y2*h/240)+margin)))
        
        if not search_areas: search_areas = [(0, 0, w, h)]

        new_boxes = []
        for (sx1, sy1, sx2, sy2) in search_areas:
            roi = frame_to_proc[sy1:sy2, sx1:sx2]
            if roi.size == 0: continue
            
            # [2단계] 최종 병기 전처리 (확대 + 멀티 필터링)
            roi_zoom = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(roi_zoom, cv2.COLOR_BGR2GRAY)
            
            # 1. 샤프닝 (테두리 강조)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            
            # 2. 모폴로지 닫기 (끊긴 글자 잇기)
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            closed = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, morph_kernel)
            
            # 3. 적응형 이진화
            processed = cv2.adaptiveThreshold(closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 9)

            # [3단계] 극한의 문자인식 (민감도 최상)
            res = reader.readtext(processed, rotation_info=[90, 270], paragraph=True, 
                                 low_text=0.15, link_threshold=0.05, mag_ratio=3.0)
            
            captured_texts = []
            for (bbox, text) in res:
                clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(clean) > 0:
                    captured_texts.append(clean)
                    (tl, tr, br, bl) = bbox
                    new_boxes.append(((int(tl[0]/2 + sx1), int(tl[1]/2 + sy1)), 
                                     (int(br[0]/2 + sx1), int(br[1]/2 + sy1)), clean))

            # 조각 모음 및 패턴 검증
            full_text = "".join(captured_texts)
            code = check_bic_code(full_text)
            if not code:
                for t in captured_texts:
                    if check_bic_code(t): code = check_bic_code(t); break

            if code:
                ccr_history.append(code)
                print(f"★ 정밀 분석 포착: {code}")

        display_boxes = new_boxes

        if len(ccr_history) > 12: ccr_history.pop(0)
        if ccr_history:
            most = Counter(ccr_history).most_common(1)[0]
            if most[1] >= 2:
                if confirmed_ccr != most[0]:
                    confirmed_ccr = most[0]
                    save_log('CCR', confirmed_ccr, 0.95)
                    # 성공 스냅샷 저장
                    cv2.imwrite(f"captures/DETECTED_{datetime.now().strftime('%H%M%S')}_{confirmed_ccr}.jpg", frame_to_proc)

        time.sleep(0.1)

def main():
    global latest_frame, is_running
    threading.Thread(target=ai_worker, daemon=True).start()

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened(): return

    print("Nova Gate Monitor Final Edition 가동!")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(RTSP_URL)
            time.sleep(1); continue
        
        latest_frame = frame.copy()
        h, w = frame.shape[:2]

        if confirmed_ccr:
            cv2.rectangle(frame, (10, h-80), (550, h-10), (0, 0, 0), -1)
            cv2.putText(frame, f"FINAL CCR: {confirmed_ccr}", (20, h-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        for (pt1, pt2, txt) in display_boxes:
            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            cv2.putText(frame, txt, (pt1[0], pt1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Nova Real-time Gate Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False; break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
