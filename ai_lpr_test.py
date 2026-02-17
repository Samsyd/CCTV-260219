import cv2
import easyocr
import time
import os
from ultralytics import YOLO
import re
from collections import Counter
import csv
from datetime import datetime

def save_log(log_type, text, prob):
    """일자별 CSV 로그 저장"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
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

    print("Nova Pro AI Engine (LPR+CCR) 초기화 중...")
    yolo_model = YOLO('yolov8n.pt') 
    reader = easyocr.Reader(['en'], gpu=False)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    
    last_ocr_time = time.time()
    lpr_history = []
    ccr_history = []
    lpr_confirmed = ""
    ccr_confirmed = ""
    display_results = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        # [1단계] 차량 탐지
        img_small = cv2.resize(frame, (320, 240))
        yolo_results = yolo_model(img_small, verbose=False, classes=[2, 7]) 
        
        detected_vehicles = []
        scale_x, scale_y = frame.shape[1]/320, frame.shape[0]/240
        
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                detected_vehicles.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

        # [2단계 & 3단계] 전략적 ROI 추출 및 문자인식
        current_time = time.time()
        if detected_vehicles and (current_time - last_ocr_time > 0.8):
            display_results = []
            # 가장 큰 차량 대상
            vx1, vy1, vx2, vy2 = max(detected_vehicles, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            vh = vy2 - vy1
            
            # ROI 설정: 상단 60%(CCR), 하단 40%(LPR)
            roi_configs = [
                {'name': 'CCR', 'area': (vy1, vy1 + int(vh*0.6), vx1, vx2), 'color': (0, 0, 255)},
                {'name': 'LPR', 'area': (vy1 + int(vh*0.6), vy2, vx1, vx2), 'color': (255, 0, 0)}
            ]

            for cfg in roi_configs:
                ry1, ry2, rx1, rx2 = cfg['area']
                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size == 0: continue
                
                # 시각화 (검색 영역 표시)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (100, 100, 100), 1)
                
                res = reader.readtext(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                for (bbox, text, prob) in res:
                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(clean_text) >= 4 and prob > 0.4:
                        (tl, tr, br, bl) = bbox
                        display_results.append({
                            'box': ((int(tl[0]+rx1), int(tl[1]+ry1)), (int(br[0]+rx1), int(br[1]+ry1))),
                            'text': clean_text, 'type': cfg['name'], 'color': cfg['color'], 'prob': prob
                        })
                        
                        # 히스토리 업데이트 및 투표
                        hist = ccr_history if cfg['name'] == 'CCR' else lpr_history
                        hist.append(clean_text)
                        if len(hist) > 10: hist.pop(0)
                        
                        counts = Counter(hist)
                        most = counts.most_common(1)[0]
                        if most[1] >= 3:
                            if cfg['name'] == 'CCR': 
                                if ccr_confirmed != most[0]:
                                    ccr_confirmed = most[0]
                                    save_log('CCR', ccr_confirmed, prob)
                            else:
                                if lpr_confirmed != most[0]:
                                    lpr_confirmed = most[0]
                                    save_log('LPR', lpr_confirmed, prob)
            
            last_ocr_time = current_time

        # 결과 렌더링
        if lpr_confirmed or ccr_confirmed:
            cv2.rectangle(frame, (10, 10), (550, 110), (0, 0, 0), -1)
            cv2.putText(frame, f"LPR: {lpr_confirmed}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 50, 50), 2)
            cv2.putText(frame, f"CCR: {ccr_confirmed}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)

        for res in display_results:
            cv2.rectangle(frame, res['box'][0], res['box'][1], res['color'], 2)
            cv2.putText(frame, res['text'], (res['box'][0][0], res['box'][0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, res['color'], 1)

        cv2.imshow("Nova LPR & CCR Master", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": main()
