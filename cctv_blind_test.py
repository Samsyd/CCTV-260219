import cv2
import threading

def test_stream(ip, ports=[554, 80, 8000, 8899]):
    # 흔히 사용되는 RTSP 경로 패턴들
    patterns = [
        "", # 기본 경로
        "/stream1",
        "/stream2",
        "/live/ch0",
        "/onvif1",
        "/Streaming/Channels/101", # Hikvision 스타일
        "/cam/realmonitor?channel=1&subtype=0" # Dahua 스타일
    ]
    
    # 한화(Hanwha/Wisenet) 및 범용 기본 자격 증명 대폭 강화
    credentials = [
        None, 
        ("admin", "4321"), # 한화 구형 기본
        ("admin", "admin1234"), # 한화 신형 기본
        ("admin", "pass1234"),
        ("admin", "admin"),
        ("admin", "12345"),
        ("admin", "123456"),
        ("admin", "888888"),
        ("admin", "password"),
        ("root", "root"),
        ("admin", "1111"),
        ("admin", "0000")
    ]

    print(f"\n[{ip}] 테스트 시작...")

    for cred in credentials:
        for port in ports:
            for path in patterns:
                if cred:
                    user, pw = cred
                    url = f"rtsp://{user}:{pw}@{ip}:{port}{path}"
                else:
                    url = f"rtsp://{ip}:{port}{path}"
                
                # print(f"시도 중: {url}") # 너무 많이 찍히면 주석 처리
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1000) # 1초 타임아웃

                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"!!! 연결 성공 !!! -> {url}")
                        cap.release()
                        return url
                    cap.release()
    
    print(f"[{ip}] 연결 가능한 스트림을 찾지 못했습니다.")
    return None

def main():
    # 스캐너에서 발견된 IP 리스트
    target_ips = ["192.168.0.4", "192.168.0.6", "192.168.0.23", "192.168.0.100"] 
    
    print("발견된 IP들에 대해 무작위 접속 테스트를 시작합니다.")
    print("영상이 확인되면 해당 주소를 알려드립니다.")

    for ip in target_ips:
        success_url = test_stream(ip)
        if success_url:
            print(f"\n최종 확인된 주소: {success_url}")
            # 첫 번째 성공한 영상을 띄워봅니다.
            cap = cv2.VideoCapture(success_url)
            while True:
                ret, frame = cap.read()
                if not ret: break
                cv2.imshow(f"Success: {ip}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            break # 일단 하나만 성공하면 중단 (샘의 확인을 위해)

if __name__ == "__main__":
    main()
