import cv2
import threading
import time

# 테스트할 RTSP 주소 리스트 (여기에 성공한 주소들을 넣으면 됩니다)
# 우선 샘이 발견한 IP들 기반으로 예시 주소를 넣었습니다.
STREAM_URLS = [
    "rtsp://192.168.0.4:554",
    "rtsp://192.168.0.6:554",
    "rtsp://192.168.0.23:554",
    "rtsp://192.168.0.100:554"
]

class VideoStream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                # 화면 크기를 동일하게 맞춤 (예: 640x480)
                self.frame = cv2.resize(frame, (640, 480))
            else:
                self.stopped = True

    def stop(self):
        self.stopped = True
        self.cap.release()

def main():
    streams = []
    for url in STREAM_URLS:
        print(f"스트림 시작 시도: {url}")
        stream = VideoStream(url).start()
        streams.append(stream)

    print("멀티 뷰어를 실행합니다. 'q'를 누르면 종료됩니다.")

    while True:
        frames = []
        for s in streams:
            if s.frame is not None:
                frames.append(s.frame)
            else:
                # 영상이 아직 로드되지 않았을 때 보여줄 검은 화면
                black_frame = bytes(640 * 480 * 3)
                import numpy as np
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

        if len(frames) >= 4:
            # 2x2 격자 구성
            top_row = cv2.hconcat([frames[0], frames[1]])
            bottom_row = cv2.hconcat([frames[2], frames[3]])
            combined = cv2.vconcat([top_row, bottom_row])
            
            cv2.imshow("CCTV Multi-Viewer (2x2)", combined)
        elif len(frames) > 0:
            # 개수가 부족하면 그냥 첫 번째 영상만 표시
            cv2.imshow("CCTV Multi-Viewer", frames[0])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for s in streams:
        s.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
