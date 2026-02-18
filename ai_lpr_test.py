import cv2
import easyocr
import time
import os
import re
from collections import Counter
import csv
from datetime import datetime
import numpy as np
import threading

# 전역 변수
latest_frame = None
confirmed_ccr = ""
is_running = True
display_results = []

RTSP_URL = "rtsp://admin:4321@192.168.0.20:554/profile2/media.smp"

def save_log(log_type, text, prob):
    log_dir = 'logs'
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    filename = f"{log_dir}/{datetime.now().strftime('%Y-%m-%d')}_gate_log.csv"
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([datetime.now().strftime("%H:%M:%S"), log_type, text, f"{prob:.2f}"])

def check_bic_structure(text):
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    match = re.search(r'[A-Z]{4}\d{6,7}', clean)
    return match.group() if match else None

def ai_worker():
    global latest_frame, confirmed_ccr, display_results
    print("Nova Logic-Hunter 가동 (AI 사물인식 의존성 제거)...")
    
    reader = easyocr.Reader(['en'], gpu=False)
    ccr_history = []
    
    # 배경 차분기 초기화 (움직임 감지용)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    while is_running:
        if latest_frame is None:
            time.sleep(0.1); continue

        frame_to_proc = latest_frame.copy()
        h, w = frame_to_proc.shape[:2]
        
        # [1단계] 논리적 물체 감지 (움직임이 큰 영역 확보)
        fgmask = fgbg.apply(cv2.resize(frame_to_proc, (320, 240)))
        motion_ratio = np.count_nonzero(fgmask) / fgmask.size
        
        # 화면의 5% 이상 변화가 있으면 물체가 있다고 판단
        if motion_ratio > 0.05:
            # [2단계] 전처리 극대화 (화면 전체를 정밀 스캔 구역으로 설정)
            # 특히 컨테이너 번호가 자주 나타나는 우측 상단/중앙 가중치
            roi = frame_to_proc[int(h*0.1):int(h*0.9), int(w*0.3):int(w*0.95)]
            
            zoom = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # [3단계] 극한의 조각 수집 및 재조립
            res = reader.readtext(gray, rotation_info=[90, 270], paragraph=False, mag_ratio=3.0)
            
            fragments = []
            for (bbox, text, prob) in res:
                clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(clean) > 0:
                    fragments.append(clean)
            
            # 조립 및 패턴 검증
            full_raw = "".join(fragments)
            code = check_bic_structure(full_raw)
            
            if not code:
                # 낱글자 조합 시도
                for f in fragments:
                    if check_bic_structure(f): code = check_bic_structure(f); break
            
            if code:
                ccr_history.append(code)
                print(f"★ 실전 포착 성공: {code}")

        # 신뢰도 투표
        if len(ccr_history) > 10: ccr_history.pop(0)
        if ccr_history:
            most = Counter(ccr_history).most_common(1)[0]
            if most[1] >= 2:
                if confirmed_ccr != most[0]:
                    confirmed_ccr = most[0]
                    save_log('CCR_REAL', confirmed_ccr, 0.9)
                    cv2.imwrite(f"captures/REAL_SUCCESS_{datetime.now().strftime('%H%M%S')}.jpg", frame_to_proc)

        time.sleep(0.5)

def main():
    global latest_frame, is_running
    threading.Thread(target=ai_worker, daemon=True).start()
    cap = cv2.VideoCapture(RTSP_URL)
    
    print("Nova Gate-Master 실전 모드 가동 중... 'q' 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(RTSP_URL); time.sleep(1); continue
        
        latest_frame = frame.copy()
        h, w = frame.shape[:2]

        if confirmed_ccr:
            cv2.rectangle(frame, (10, h-80), (600, h-10), (0, 0, 0), -1)
            cv2.putText(frame, f"GATE CCR: {confirmed_ccr}", (20, h-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

        cv2.imshow("Nova Intelligent Gate Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False; break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
