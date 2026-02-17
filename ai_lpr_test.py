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

def main():
    if not os.path.exists('captures'): os.makedirs('captures')

    print("Nova Pattern-Hunter (High-Resolution Boost) 초기화 중...")
    yolo_model = YOLO('yolov8n.pt')
    reader = easyocr.Reader(['en'], gpu=False)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    
    last_ocr_time = time.time()
    ccr_history = []
    ccr_confirmed = ""
    display_boxes = []

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
        if current_time - last_ocr_time > 0.8:
            display_boxes = []
            search_areas = detected_targets if detected_targets else [(0, 0, w, h)]
            
            for (sx1, sy1, sx2, sy2) in search_areas:
                # ROI 확대 분석 (1.5배)
                roi = frame[max(0, sy1-30):min(h, sy2+30), max(0, sx1-30):min(w, sx2+30)]
                if roi.size == 0: continue
                
                h_r, w_r = roi.shape[:2]
                roi_zoom = cv2.resize(roi, (int(w_r*1.5), int(h_r*1.5)), interpolation=cv2.INTER_CUBIC)
                
                # 샤프닝 커널 적용
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(roi_zoom, -1, kernel)
                
                # 전처리: 대비 강화
                gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
                processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 5)

                res = reader.readtext(processed, rotation_info=[90, 270], paragraph=True, mag_ratio=2.0)
                
                for (bbox, text) in res:
                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if re.search(r'[A-Z]{3,4}\d{5,7}', clean_text) or len(clean_text) >= 10:
                        ccr_history.append(clean_text)
                        (tl, tr, br, bl) = bbox
                        orig_tl = (int(tl[0]/1.5 + max(0, sx1-30)), int(tl[1]/1.5 + max(0, sy1-30)))
                        orig_br = (int(br[0]/1.5 + max(0, sx1-30)), int(br[1]/1.5 + max(0, sy1-30)))
                        display_boxes.append((orig_tl, orig_br, clean_text))
                        print(f"★ 패턴 포착: {clean_text}")

            if len(ccr_history) > 15: ccr_history.pop(0)
            if ccr_history:
                counts = Counter(ccr_history)
                most = counts.most_common(1)[0]
                if most[1] >= 2:
                    if ccr_confirmed != most[0]:
                        ccr_confirmed = most[0]
                        save_log('CCR', ccr_confirmed, 0.9)
                        cv2.imwrite(f"captures/CCR_BOOST_{datetime.now().strftime('%H%M%S')}.jpg", frame)

            last_ocr_time = current_time

        if ccr_confirmed:
            cv2.rectangle(frame, (10, h-70), (550, h-10), (0, 0, 0), -1)
            cv2.putText(frame, f"CONFIRMED CCR: {ccr_confirmed}", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        for (pt1, pt2, txt) in display_boxes:
            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            cv2.putText(frame, txt, (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Nova CCR Boost Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
