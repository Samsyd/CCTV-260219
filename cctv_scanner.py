import socket
import threading
from queue import Queue

def scan_port(ip, port, found_devices):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((ip, port)) == 0:
                found_devices.append((ip, port))
                print(f"[찾음] {ip}:{port} - 카메라 서비스 포트가 열려 있습니다.")
    except:
        pass

def main():
    # 샘의 네트워크 대역에 맞게 수정하세요 (보통 192.168.0 또는 192.168.1)
    base_ip = "192.168.0." 
    # 카메라가 주로 사용하는 포트들
    ports = [80, 554, 8000, 8899]
    
    found_devices = []
    threads = []

    print(f"{base_ip}1 ~ 254 대역에서 카메라를 검색합니다. 잠시만 기다려주세요...")

    for i in range(1, 255):
        ip = base_ip + str(i)
        for port in ports:
            t = threading.Thread(target=scan_port, args=(ip, port, found_devices))
            t.start()
            threads.append(t)
            
            # 너무 많은 스레드가 한꺼번에 돌지 않도록 조절
            if len(threads) > 100:
                for th in threads:
                    th.join()
                threads = []

    for t in threads:
        t.join()

    print("\n--- 검색 결과 ---")
    if not found_devices:
        print("검색된 장치가 없습니다. IP 대역(base_ip)이 맞는지 확인해 주세요.")
    else:
        for ip, port in found_devices:
            print(f"장치 발견: {ip} (Port: {port})")

if __name__ == "__main__":
    main()
