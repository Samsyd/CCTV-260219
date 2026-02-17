import cv2
import easyocr
import time
import os
import re
from collections import Counter
import csv
from datetime import datetime

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

    print("Nova Gate-Master (CCR Priority) 엔진 초기화 중...")
    # 컨테이너 번호(영문/숫자) 인식을 위해 'en' 모드 최적화
    reader = easyocr.Reader(['en'], gpu=False)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    
    last_ocr_time = time.time()
    ccr_history = []
    ccr_confirmed = ""
    
    print("시스템 가동: 게이트 모드 (전체 화면 감시 및 우측 집중 분석)")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]

        # [1단계 & 2단계] 전략적 구역 분할 (YOLO 없이 직접 ROI 지정)
        # 컨테이너 게이트 특성상 화면 우측 상단/중앙을 CCR 타겟으로 설정
        ccr_roi_x1 = int(w * 0.5) # 화면 우측 절반
        ccr_roi_y1 = int(h * 0.1) # 상단 10%부터
        ccr_roi_y2 = int(h * 0.9) # 하단 90%까지
        
        # 가이드라인 표시 (샘이 확인하기 쉽게 노란색 박스 표시)
        cv2.rectangle(frame, (ccr_roi_x1, ccr_roi_y1), (w-10, ccr_roi_y2), (0, 255, 255), 2)
        cv2.putText(frame, "CCR SCAN AREA (RIGHT)", (ccr_roi_x1, ccr_roi_y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # [3단계] 정밀 문자인식 (0.7초 주기로 강력 실행)
        current_time = time.time()
        if current_time - last_ocr_time > 0.7:
            # 우측 ROI 잘라내기
            roi = frame[ccr_roi_y1:ccr_roi_y2, ccr_roi_x1:w]
            
            # 이미지 전처리: 선명도 및 대비 극대화
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            processed_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # 모든 방향(가로/세로) 문자 스캔
            res = reader.readtext(processed_roi, rotation_info=[90, 270], paragraph=True)
            
            for (bbox, text) in res:
                # 텍스트 정제 (대문자, 숫자만)
                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                # 컨테이너 번호(BIC Code) 패턴 매칭: [영문3-4자] + [숫자 5-7자]
                if re.search(r'[A-Z]{3,4}\d{5,7}', clean_text) or len(clean_text) >= 10:
                    ccr_history.append(clean_text)
                    print(f"★ 감지됨: {clean_text}")

            # 최근 인식 결과 중 최다 득표 결과 확정
            if len(ccr_history) > 10: ccr_history.pop(0)
            if ccr_history:
                counts = Counter(ccr_history)
                most = counts.most_common(1)[0]
                if most[1] >= 2: # 2회 반복 시 확정
                    if ccr_confirmed != most[0]:
                        ccr_confirmed = most[0]
                        save_log('CCR', ccr_confirmed, 0.9)
                        # 스냅샷 저장
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"captures/CCR_{timestamp}_{ccr_confirmed}.jpg", frame)

            last_ocr_time = current_time

        # 결과 화면 출력
        if ccr_confirmed:
            cv2.rectangle(frame, (10, h-80), (600, h-10), (0, 0, 0), -1)
            cv2.putText(frame, f"LAST CONFIRMED CCR: {ccr_confirmed}", (20, h-35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        cv2.imshow("Nova Gate-Master Master", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
