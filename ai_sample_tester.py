import cv2
import easyocr
import re
import numpy as np
import os

def extract_bic_parts(text):
    """글자 조각에서 영문4자 또는 숫자6-7자 추출"""
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    # 영문 4자 후보
    alpha = re.search(r'[A-Z]{4}', clean)
    # 숫자 6-7자 후보
    digit = re.search(r'\d{6,7}', clean)
    return alpha.group() if alpha else None, digit.group() if digit else None

def process_sample(image_path, reader, output_name):
    if not os.path.exists(image_path): return
    frame = cv2.imread(image_path)
    if frame is None: return
    h, w = frame.shape[:2]
    
    # 전처리 극대화
    zoom = cv2.resize(frame, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 9)

    print(f"[{os.path.basename(image_path)}] 조각 모음 분석 중...")
    res = reader.readtext(processed, rotation_info=[90, 270], paragraph=False)
    
    found_alphas = []
    found_digits = []
    
    for (bbox, text, prob) in res:
        a, d = extract_bic_parts(text)
        if a: found_alphas.append(a)
        if d: found_digits.append(d)
        print(f"감지 조각: {text} -> (영문:{a}, 숫자:{d})")

    # 조각 합치기 시도 (가장 유력한 것들끼리)
    final_code = None
    if found_alphas and found_digits:
        final_code = found_alphas[0] + found_digits[0]
    elif found_digits:
        final_code = "UNKNOWN" + found_digits[0]

    # 결과 표시
    cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
    status = f"FINAL RESULT: {final_code}" if final_code else "FAILED"
    color = (0, 255, 0) if final_code and "UNKNOWN" not in final_code else (0, 0, 255)
    cv2.putText(frame, status, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imwrite(f"sample_result_{output_name}.jpg", frame)
    print(f"★ 조각 모음 결과: {final_code}")

def main():
    samples = ["sample_0.jpg", "sample_1.jpg"]
    reader = easyocr.Reader(['en'], gpu=False)
    for i, s in enumerate(samples):
        process_sample(s, reader, i)

if __name__ == "__main__": main()
