import cv2
import os
import time

def test_specific_camera(ip, user, password):
    # 한화(Hanwha/Wisenet) 카메라의 주요 RTSP 경로 패턴
    paths = [
        "/profile2/media.smp",
        "/live/ch0",
        "/onvif-media/media.smp",
        "/strm/v1/live/1",
        "" # 기본 경로
    ]
    
    print(f"\n[{ip}] 연결 시도 중 (ID: {user} / PW: {password})")
    
    for path in paths:
        url = f"rtsp://{user}:{password}@{ip}:554{path}"
        # print(f"시도: {url}")
        
        cap = cv2.VideoCapture(url)
        # 타임아웃 설정 (1초)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1000)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"!!! 연결 성공 !!!")
                print(f"최종 주소: {url}")
                
                # 성공 스냅샷 저장
                if not os.path.exists('captures'): os.makedirs('captures')
                cv2.imwrite(f"captures/camera_connect_success.jpg", frame)
                
                cap.release()
                return url
            cap.release()
            
    print(f"[{ip}] 모든 경로에서 연결에 실패했습니다. 비밀번호나 경로를 다시 확인해야 합니다.")
    return None

if __name__ == "__main__":
    target_ip = "192.168.0.20"
    admin_user = "admin"
    admin_pw = "4321" # 샘이 알려주신 초기 비밀번호
    
    success_url = test_specific_camera(target_ip, admin_user, admin_pw)
    
    if success_url:
        print("\n이제 이 주소를 우리 AI 엔진에 연결할 수 있습니다!")
