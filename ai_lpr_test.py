import cv2
import easyocr
import time
import os
from ultralytics import YOLO
import re
from collections import Counter
import csv
from datetime import datetime
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

    print("Nova Ultra CCR & LPR Engine 초기화 중...")
    yolo_model = YOLO('yolov8n.pt') 
    reader = easyocr.Reader(['en'], gpu=False)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    
    last_ocr_time = time.time()
    lpr_history, ccr_history = [], []
    lpr_confirmed, ccr_confirmed = "", ""
    display_results = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        # [1단계] 고감도 물체 탐지 (컨테이너 포함)
        img_small = cv2.resize(frame, (320, 240))
        # conf=0.2로 낮추어 희미한 컨테이너 박스도 포착
        yolo_results = yolo_model(img_small, verbose=False, conf=0.2, classes=[2, 5, 6, 7]) 
        
        detected_targets = []
        scale_x, scale_y = frame.shape[1]/320, frame.shape[0]/240
        
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                is_large = int(box.cls[0]) in [5, 6, 7]
                detected_targets.append({'box': (x1, y1, x2, y2), 'is_large': is_large})
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

        # [2단계 & 3단계] 전략적 우측/하단 분석
        current_time = time.time()
        if detected_targets and (current_time - last_ocr_time > 0.7):
            display_results = []
            target = max(detected_targets, key=lambda b: (b['box'][2]-b['box'][0])*(b['box'][3]-b['box'][1]))
            tx1, ty1, tx2, ty2 = target['box']
            tw, th = tx2 - tx1, ty2 - ty1

            # 전략: 컨테이너(대형)는 우측 40%, 차량은 하단 40%
            if target['is_large']:
                # 컨테이너 번호는 주로 우측(Right side)에 위치
                roi_configs = [{'name': 'CCR', 'area': (ty1, ty2, tx1 + int(tw*0.6), tx2), 'color': (0, 0, 255)}]
            else:
                roi_configs = [{'name': 'LPR', 'area': (ty1 + int(th*0.6), ty2, tx1, tx2), 'color': (255, 0, 0)}]

            for cfg in roi_configs:
                ry1, ry2, rx1, rx2 = cfg['area']
                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size == 0: continue
                
                # 시각화: 현재 분석 중인 영역 (점선 느낌)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                
                # 컨테이너는 정밀 전처리 및 세로 인식 적용
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                if cfg['name'] == 'CCR':
                    gray_roi = cv2.equalizeHist(gray_roi) # 대비 강제 최적화
                    res = reader.readtext(gray_roi, rotation_info=[90, 270], paragraph=True)
                else:
                    res = reader.readtext(gray_roi)

                for (bbox, text, prob) in (res if cfg['name'] != 'CCR' else [(r[0], r[1], 0.8) for r in res]):
                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(clean_text) >= 4:
                        (tl, tr, br, bl) = bbox
                        display_results.append({
                            'box': ((int(tl[0]+rx1), int(tl[1]+ry1)), (int(br[0]+rx1), int(br[1]+ry1))),
                            'text': clean_text, 'type': cfg['name'], 'color': cfg['color']
                        })
                        
                        hist = ccr_history if cfg['name'] == 'CCR' else lpr_history
                        hist.append(clean_text)
                        if len(hist) > 10: hist.pop(0)
                        
                        counts = Counter(hist)
                        most = counts.most_common(1)[0]
                        if most[1] >= 2: # 인식률 향상을 위해 투표 기준을 2회로 완화
                            if cfg['name'] == 'CCR':
                                if ccr_confirmed != most[0]:
                                    ccr_confirmed = most[0]; save_log('CCR', ccr_confirmed, 0.8)
                            else:
                                if lpr_confirmed != most[0]:
                                    lpr_confirmed = most[0]; save_log('LPR', lpr_confirmed, 0.8)
            last_ocr_time = current_time

        # 결과 표시
        if lpr_confirmed or ccr_confirmed:
            cv2.rectangle(frame, (10, 10), (500, 110), (0, 0, 0), -1)
            if lpr_confirmed: cv2.putText(frame, f"LPR: {lpr_confirmed}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 50, 50), 2)
            if ccr_confirmed: cv2.putText(frame, f"CCR: {ccr_confirmed}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)

        for res in display_results:
            cv2.rectangle(frame, res['box'][0], res['box'][1], res['color'], 2)
            cv2.putText(frame, res['text'], (res['box'][0][0], res['box'][0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, res['color'], 1)

        cv2.imshow("Nova Ultra Master", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
