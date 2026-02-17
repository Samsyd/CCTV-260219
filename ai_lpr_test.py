import cv2
import easyocr
import time
import os
import re
from collections import Counter
import csv
from datetime import datetime
from ultralytics import YOLO

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

    print("Nova Pattern-Hunter (BIC Code Priority) 엔진 초기화 중...")
    yolo_model = YOLO('yolov8n.pt')
    reader = easyocr.Reader(['en'], gpu=False)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    
    last_ocr_time = time.time()
    ccr_history = []
    ccr_confirmed = ""
    display_boxes = []

    print("시스템 가동: 패턴 기반 인식 모드 (규격: 영문4자 + 숫자7자)")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]

        # [1단계] 광범위 객체 탐지
        img_small = cv2.resize(frame, (320, 240))
        yolo_results = yolo_model(img_small, verbose=False, conf=0.25, classes=[2, 5, 6, 7])
        
        detected_targets = []
        scale_x, scale_y = w/320, h/240
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_targets.append((int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)))

        # [2단계 & 3단계] 패턴 매칭 기반 OCR (0.7초 주기)
        current_time = time.time()
        if current_time - last_ocr_time > 0.7:
            display_boxes = []
            
            # 탐지된 물체가 있으면 그 영역을, 없으면 화면 전체를 분석
            search_areas = detected_targets if detected_targets else [(0, 0, w, h)]
            
            for (sx1, sy1, sx2, sy2) in search_areas:
                # 마진 추가
                sx1, sy1 = max(0, sx1-20), max(0, sy1-20)
                sx2, sy2 = min(w, sx2+20), min(h, sy2+20)
                
                roi = frame[sy1:sy2, sx1:sx2]
                if roi.size == 0: continue
                
                # 이미지 전처리: 패턴 부각을 위한 대비 최적화
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                processed_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                # 전방위 회전 스캔 (세로 번호 대응)
                res = reader.readtext(processed_roi, rotation_info=[90, 270], paragraph=True)
                
                for (bbox, text) in res:
                    # 영문/숫자만 남김
                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    
                    # [핵심] 컨테이너 번호 규격 필터링: 영문 4자 + 숫자 6~7자
                    # 패턴: ^[A-Z]{4}[0-9]{6,7}$
                    if re.search(r'[A-Z]{4}[0-9]{6,7}', clean_text):
                        ccr_history.append(clean_text)
                        # 좌표 변환 및 표시용 저장
                        (tl, tr, br, bl) = bbox
                        display_boxes.append(((int(tl[0]+sx1), int(tl[1]+sy1)), 
                                            (int(br[0]+sx1), int(br[1]+sy1)), clean_text))
                        print(f"★ 규격 일치 번호 감지: {clean_text}")

            # 투표 시스템으로 신뢰도 확보
            if len(ccr_history) > 15: ccr_history.pop(0)
            if ccr_history:
                counts = Counter(ccr_history)
                most = counts.most_common(1)[0]
                if most[1] >= 2: # 2회 반복 시 확정
                    if ccr_confirmed != most[0]:
                        ccr_confirmed = most[0]
                        save_log('CCR', ccr_confirmed, 0.95)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"captures/CCR_PATTERN_{timestamp}_{ccr_confirmed}.jpg", frame)

            last_ocr_time = current_time

        # 시각화
        if ccr_confirmed:
            cv2.rectangle(frame, (10, h-70), (550, h-10), (0, 0, 0), -1)
            cv2.putText(frame, f"CONFIRMED CCR: {ccr_confirmed}", (20, h-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        for (pt1, pt2, txt) in display_boxes:
            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            cv2.putText(frame, txt, (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Nova Pattern-Hunter Master", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
