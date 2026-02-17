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
    """컨테이너 BIC Code 표준 규격 확인 (영문4 + 숫자7)"""
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    # 1. 완벽한 패턴 (영문4 + 숫자7)
    if re.search(r'[A-Z]{4}\d{7}', clean):
        return re.search(r'[A-Z]{4}\d{7}', clean).group()
    # 2. 부분 패턴 (영문4 + 숫자6) - 체크디지트 누락 대비
    if re.search(r'[A-Z]{4}\d{6}', clean):
        return re.search(r'[A-Z]{4}\d{6}', clean).group()
    return None

def main():
    if not os.path.exists('captures'): os.makedirs('captures')

    print("Nova Pattern-Hunter Pro (Spatial Merging Engine) 초기화 중...")
    yolo_model = YOLO('yolov8n.pt')
    # 영문 인식을 위해 최적화
    reader = easyocr.Reader(['en'], gpu=False)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    
    last_ocr_time = time.time()
    ccr_history = []
    ccr_confirmed = ""
    display_boxes = []

    print("시스템 가동: 공간 병합 인식 모드 (TEMU/MSCU 사례 대응)")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        img_small = cv2.resize(frame, (320, 240))
        yolo_results = yolo_model(img_small, verbose=False, conf=0.25, classes=[2, 5, 6, 7])
        
        detected_targets = []
        scale_x, scale_y = w/320, h/240
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_targets.append((int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)))

        current_time = time.time()
        if current_time - last_ocr_time > 0.7:
            display_boxes = []
            search_areas = detected_targets if detected_targets else [(0, 0, w, h)]
            
            for (sx1, sy1, sx2, sy2) in search_areas:
                roi = frame[max(0, sy1-30):min(h, sy2+30), max(0, sx1-30):min(w, sx2+30)]
                if roi.size == 0: continue
                
                # [성능 강화] 1.5배 확대 및 샤프닝
                roi_zoom = cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(roi_zoom, -1, kernel)
                gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
                processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 5)

                # [성능 강화] 세로/가로 동시 스캔
                res = reader.readtext(processed, rotation_info=[90, 270], paragraph=False)
                
                # --- 공간 병합 로직 (Spatial Merging) ---
                raw_texts = []
                for (bbox, text, prob) in res:
                    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(clean) > 0:
                        # 좌표 복원
                        (tl, tr, br, bl) = bbox
                        cx = (tl[0] + br[0]) / 2
                        cy = (tl[1] + br[1]) / 2
                        raw_texts.append({'text': clean, 'cx': cx, 'cy': cy, 'bbox': bbox})

                # 1. 가로/세로 근접 텍스트 병합 시도
                combined_text = "".join([t['text'] for t in raw_texts])
                
                # 2. 개별 텍스트 및 병합 텍스트에서 BIC Code 찾기
                candidates = []
                if check_bic_code(combined_text):
                    candidates.append(check_bic_code(combined_text))
                for t in raw_texts:
                    if check_bic_code(t['text']):
                        candidates.append(check_bic_code(t['text']))

                for code in candidates:
                    ccr_history.append(code)
                    print(f"★ 컨테이너 번호 포착: {code}")
                    # 일단 첫 번째 후보 표시
                    display_boxes.append(((sx1, sy1), (sx2, sy2), code))

            # 투표 시스템
            if len(ccr_history) > 15: ccr_history.pop(0)
            if ccr_history:
                counts = Counter(ccr_history)
                most = counts.most_common(1)[0]
                if most[1] >= 2:
                    if ccr_confirmed != most[0]:
                        ccr_confirmed = most[0]
                        save_log('CCR', ccr_confirmed, 0.95)
                        cv2.imwrite(f"captures/CCR_FINAL_{datetime.now().strftime('%H%M%S')}.jpg", frame)

            last_ocr_time = current_time

        # UI 출력
        if ccr_confirmed:
            cv2.rectangle(frame, (10, h-70), (600, h-10), (0, 0, 0), -1)
            cv2.putText(frame, f"CONFIRMED CCR: {ccr_confirmed}", (20, h-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        cv2.imshow("Nova Spatial-Hunter Master", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
