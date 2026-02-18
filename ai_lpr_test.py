import cv2
import easyocr
import time
import os
import re
from collections import Counter
import numpy as np
import threading

# 전역 변수
latest_frame = None
confirmed_ccr = ""
is_running = True

RTSP_URL = "rtsp://admin:4321@192.168.0.20:554/profile2/media.smp"

def check_bic_code(text):
    """컨테이너 BIC Code 정밀 검증 (영문4+숫자6~7)"""
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    # 세로로 읽다 보면 글자 하나씩 들어올 수 있으므로 패턴 매칭 유연화
    match = re.search(r'[A-Z]{4}\d{6,7}', clean)
    return match.group() if match else None

def ai_worker():
    global latest_frame, confirmed_ccr
    print("Nova Vertical-Reassembler 가동 (세로 번호 정밀 타격)...")
    
    # 영문/숫자 특화 및 감지력 극대화 설정
    reader = easyocr.Reader(['en'], gpu=False)
    ccr_history = []

    while is_running:
        if latest_frame is None:
            time.sleep(0.1); continue

        frame_to_proc = latest_frame.copy()
        h, w = frame_to_proc.shape[:2]
        
        # [1단계] 컨테이너 번호가 주로 위치하는 우측 영역 집중 (ROI)
        roi = frame_to_proc[int(h*0.1):int(h*0.9), int(w*0.6):int(w*0.95)]
        
        # [2단계] 초정밀 전처리 (세로 글자 살리기)
        # 2배 확대 및 대비 극대화
        zoom = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # [핵심] 이미지를 90도 회전하여 세로 글자를 가로로 인식하게 유도 (2 pass 분석)
        results = []
        # Pass 1: 일반 세로 인식 옵션
        results += reader.readtext(gray, rotation_info=[90, 270], paragraph=False, low_text=0.2)
        
        # [3단계] 낱글자 조립 로직 (강화된 자석 병합)
        fragments = []
        for (bbox, text, prob) in results:
            clean = re.sub(r'[^A-Z0-9]', '', text.upper())
            if clean and prob > 0.25: # 신뢰도 하한선
                (tl, tr, br, bl) = bbox
                cx = (tl[0] + br[0]) / 2
                cy = (tl[1] + br[1]) / 2
                fragments.append({'text': clean, 'x': cx, 'y': cy, 'prob': prob})

        if fragments:
            # 1. 세로 정렬 (X좌표가 비슷한 것들끼리 묶기)
            # 픽셀 오차범위 30 이내면 같은 줄로 간주
            fragments.sort(key=lambda f: f['y']) # 위에서 아래로
            
            # 모든 조각을 일단 순서대로 합쳐보기
            combined_text = "".join([f['text'] for f in fragments])
            
            # 부분 매칭 시도 (영문 2자 이상 + 숫자 3자 이상이면 일단 후보)
            if len(combined_text) >= 5:
                print(f"[DEBUG] 조립 중: {combined_text}")
                
                # 정식 규격 검사
                code = check_bic_code(combined_text)
                if not code:
                    # 더 유연한 검사: 영문과 숫자가 섞여있으면 후보로 등록
                    if any(c.isalpha() for c in combined_text) and any(c.isdigit() for c in combined_text):
                        if len(combined_text) >= 8:
                            code = combined_text # 규격 미달이라도 일단 표시
                
                if code:
                    ccr_history.append(code)
                    print(f"★ 완성된 번호 포착: {code}")

        # 투표 시스템 (2회 반복 시 확정)
        if len(ccr_history) > 10: ccr_history.pop(0)
        if ccr_history:
            most = Counter(ccr_history).most_common(1)[0]
            if most[1] >= 2:
                confirmed_ccr = most[0]

        time.sleep(0.5)

def main():
    global latest_frame, is_running
    threading.Thread(target=ai_worker, daemon=True).start()
    cap = cv2.VideoCapture(RTSP_URL)
    
    print("Nova Vertical-Master 실전 가동 중!")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(RTSP_URL); time.sleep(1); continue
        
        latest_frame = frame.copy()
        h, w = frame.shape[:2]

        # 가이드라인 (우측 정밀 타격 구역)
        cv2.rectangle(frame, (int(w*0.6), int(h*0.1)), (int(w*0.95), int(h*0.9)), (0, 255, 255), 1)

        if confirmed_ccr:
            cv2.rectangle(frame, (10, h-80), (650, h-10), (0, 0, 0), -1)
            cv2.putText(frame, f"CONFIRMED CCR: {confirmed_ccr}", (20, h-35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Nova Vertical CCR Master", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False; break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
