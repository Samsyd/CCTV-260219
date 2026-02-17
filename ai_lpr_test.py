import cv2
import easyocr
import time
from datetime import datetime
import os

def main():
    # 캡처 저장 폴더 생성
    if not os.path.exists('captures'):
        os.makedirs('captures')

    # OCR 엔진 초기화 (한국어, 영어 지원)
    print("AI 엔진을 초기화 중입니다. 잠시만 기다려 주세요...")
    reader = easyocr.Reader(['ko', 'en'], gpu=False) # CPU 모드 사용
    
    # 웹캠 연결
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("AI 인식 테스트 시작! 'q'를 누르면 종료합니다.")
    
    last_ocr_time = time.time()
    detected_text = ""

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1초마다 OCR 수행 (CPU 부하 방지)
        current_time = time.time()
        if current_time - last_ocr_time > 1.0:
            # 인식 속도 향상을 위해 흑백 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # OCR 실행
            results = reader.readtext(gray)
            
            for (bbox, text, prob) in results:
                if prob > 0.5: # 신뢰도 50% 이상만 표시
                    detected_text = text
                    print(f"인식됨: {text} (신뢰도: {prob:.2f})")
                    
                    # 스냅샷 저장 (테스트용)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"captures/detect_{timestamp}.jpg", frame)
            
            last_ocr_time = current_time

        # 화면에 결과 표시
        cv2.putText(frame, f"AI Detect: {detected_text}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Nova AI LPR/CCR Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
