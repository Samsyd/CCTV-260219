import socket
from onvif import ONVIFCamera
import zeep

def scanner():
    print("네트워크에서 ONVIF 지원 CCTV를 검색 중입니다...")
    # ONVIF 표준 포트 및 WS-Discovery는 복잡할 수 있으므로 
    # 여기서는 간단한 가이드와 함께 기본적인 장치 정보를 가져오는 구조를 제안합니다.
    
    # 실제 운영 환경에서는 'wsdiscovery' 라이브러리 등을 사용하여 
    # 멀티캐스트(239.255.255.250)를 통해 장치를 찾습니다.
    
    print("팁: 대부분의 IP 카메라는 ONVIF 서비스가 활성화되어 있어야 검색이 가능합니다.")
    print("카메라 설정 웹페이지에서 ONVIF가 켜져 있는지 확인하세요.")

def connect_camera(ip, port, user, password):
    try:
        # ONVIF 카메라 객체 생성
        mycam = ONVIFCamera(ip, port, user, password)
        
        # 기기 정보 가져오기
        device_info = mycam.devicemgmt.GetDeviceInformation()
        print(f"연결 성공!")
        print(f"제조사: {device_info.Manufacturer}")
        print(f"모델명: {device_info.Model}")
        print(f"펌웨어 버전: {device_info.FirmwareVersion}")

        # 미디어 프로필 가져오기 (RTSP 주소 획득용)
        media_service = mycam.create_media_service()
        profiles = media_service.GetProfiles()
        token = profiles[0].token

        # RTSP 스트림 주소 가져오기
        obj = media_service.create_type('GetStreamUri')
        obj.ProfileToken = token
        obj.StreamSetup = {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}
        uri_obj = media_service.GetStreamUri(obj)
        
        print(f"RTSP 스트림 주소: {uri_obj.Uri}")
        return uri_obj.Uri

    except Exception as e:
        print(f"연결 실패 ({ip}): {e}")
        return None

if __name__ == "__main__":
    scanner()
    # 테스트용 (실제 카메라 정보를 입력하여 테스트 가능)
    # target_ip = input("테스트할 카메라 IP: ")
    # connect_camera(target_ip, 80, 'admin', 'password')
