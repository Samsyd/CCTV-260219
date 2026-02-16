import cv2
import sys

def main():
    # RTSP 주소 예시 (샘의 환경에 맞게 수정 필요)
    # 형식: rtsp://username:password@ip_address:port/path
    # 예: rtsp://admin:12345@192.168.1.100:554/stream1
    
    # 테스트를 위해 웹캠(0) 또는 샘플 RTSP 주소를 입력하세요.
    rtsp_url = input("연결할 RTSP 주소를 입력하세요 (또는 웹캠 사용 시 '0' 입력): ")
    
    if rtsp_url == '0':
        rtsp_url = 0

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("에러: 카메라 또는 스트림을 열 수 없습니다.")
        return

    print("연결 성공! 영상을 보려면 'q' 키를 누르세요.")

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            print("에러: 프레임을 읽어올 수 없습니다.")
            break

        # 영상 표시
        cv2.imshow('CCTV Stream Test', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
