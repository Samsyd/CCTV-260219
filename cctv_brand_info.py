import socket
import re

def get_brand_info(ip):
    print(f"\n[{ip}] 제조사 정보 확인 시도 중...")
    
    # 1. HTTP 헤더를 통해 정보 확인 (가장 간단한 방법)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            s.connect((ip, 80))
            s.sendall(b"GET / HTTP/1.1\r\nHost: " + ip.encode() + b"\r\n\r\n")
            response = s.recv(1024).decode(errors='ignore')
            
            # Server 헤더나 특이한 문구 확인
            if "Server:" in response:
                server_info = re.search(r"Server: (.*)", response)
                if server_info:
                    print(f"서버 정보 발견: {server_info.group(1)}")
            
            if "Hikvision" in response.lower(): print("예상 제조사: Hikvision (하이킈전)")
            elif "dahua" in response.lower(): print("예상 제조사: Dahua (다후아)")
            elif "hanwha" in response.lower() or "wisenet" in response.lower(): print("예상 제조사: Hanwha (한화테크윈)")
            elif "xmeye" in response.lower(): print("예상 제조사: XMeye (중국 범용)")
    except:
        print("HTTP를 통한 정보 획득 실패.")

def main():
    target_ips = ["192.168.0.4", "192.168.0.23", "192.168.0.20", "192.168.0.100"]
    for ip in target_ips:
        get_brand_info(ip)

if __name__ == "__main__":
    main()
