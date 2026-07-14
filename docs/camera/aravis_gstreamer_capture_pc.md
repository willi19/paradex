# Paradex Aravis/GStreamer 설치 및 실행

이 문서는 capture PC의 AVI 녹화 카메라와 main PC의 timestamp 카메라를
PySpin 대신 Aravis로 실행하는 방법을 정리한다. UTG900E는 기존과 같이 main
PC에만 연결한다.

## 구성

```text
main PC
  CaptureSession
  UTG900E
  Aravis timestamp camera (frame_id + PC 수신 시각만 저장)
       |
       +-- ZMQ --> capture PC server_daemon
                        Aravis stream -> GStreamer appsrc
                        -> Bayer 변환 -> JPEG -> AVI
```

capture PC 카메라는 daemon 시작 시 연결·설정·버퍼 할당을 한 번 수행한다.
각 캡처 세션에서는 acquisition과 AVI 파이프라인만 시작/종료한다. 모든 PC와
timestamp 카메라가 acquisition 상태가 된 다음 main PC가 UTG를 켠다.

timestamp 카메라는 GStreamer를 사용하지 않고 프레임 ID와 `time.time()`만
저장한다. 이미지 변환·복사·저장은 하지 않지만, 프레임 메타데이터가 GVSP
버퍼에 포함되므로 카메라에서 main PC로 이미지 payload가 전송되는 것 자체는
필요하다.

timestamp 카메라는 모든 capture 카메라와 같은 UTG pulse를 받는 master
timeline이다. 시작할 때는 모든 capture 카메라와 timestamp 카메라를 먼저
acquisition 상태로 만든 뒤 UTG를 켠다. 종료할 때는 UTG를 먼저 꺼 마지막
trigger edge를 고정한 뒤 timestamp와 capture 카메라를 종료한다. 따라서
정상 캡처에서는 영상의 n번째 프레임을 `timestamps/timestamp.npy`의 n번째
값에 대응시킨다.

카메라마다 내부 frame counter의 시작점이 다를 수 있으므로 서로 다른 카메라의
절대 `frame_id` 값 자체가 같아야 하는 것은 아니다. 중요한 것은 세션 안에서
frame ID가 연속적이고 timestamp 배열과 각 AVI의 프레임 수가 같은 것이다.
종료 순서를 맞춘 뒤에도 개수가 다르면 해당 카메라 또는 수신/인코딩 경로에서
실제 frame drop이 발생한 것이다.

## 1. Ubuntu 패키지 설치

Aravis GI 패키지는 pip 패키지가 아니라 Ubuntu 패키지다. `flir_env` 같은
Conda 환경을 쓰더라도 아래 패키지를 OS에 먼저 설치한다.

먼저 Ubuntu 버전을 확인한다.

```bash
. /etc/os-release
echo "$VERSION_ID"
```

### Ubuntu 22.04 이상

배포판 저장소에 Aravis 0.8 GI 패키지가 있는 경우 다음을 실행한다.

```bash
sudo apt update
sudo apt install -y \
  python3-gi gir1.2-aravis-0.8 aravis-tools \
  gir1.2-gstreamer-1.0 gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  ffmpeg ethtool
```

### Ubuntu 20.04

Ubuntu 20.04(focal) 기본 저장소에는 `gir1.2-aravis-0.8`이 없고
`gir1.2-aravis-0.6`만 있다. Paradex 코드는 Aravis 0.8 API를 사용하므로
0.6 패키지를 설치하면 안 된다. GStreamer와 빌드 의존성을 먼저 설치한 뒤
공식 Aravis 0.8.20을 `/usr/local`에 설치한다.

```bash
sudo apt update
sudo apt install -y \
  python3-gi python3-venv git build-essential meson ninja-build pkg-config \
  libglib2.0-dev libxml2-dev zlib1g-dev libusb-1.0-0-dev \
  libgirepository1.0-dev gobject-introspection \
  gir1.2-gstreamer-1.0 gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  ffmpeg ethtool

git clone --branch 0.8.20 --depth 1 \
  https://github.com/AravisProject/aravis.git /tmp/aravis-0.8.20
meson setup /tmp/aravis-0.8.20/build-focal /tmp/aravis-0.8.20 \
  --prefix=/usr/local \
  -Dintrospection=enabled \
  -Dviewer=disabled \
  -Ddocumentation=disabled \
  -Dgst-plugin=disabled \
  -Dtests=false
ninja -C /tmp/aravis-0.8.20/build-focal
sudo ninja -C /tmp/aravis-0.8.20/build-focal install
sudo ldconfig
```

Paradex는 Aravis의 GStreamer `aravissrc` 플러그인을 사용하지 않는다. Aravis
stream API로 받은 버퍼를 GStreamer `appsrc`에 전달하므로 위 빌드에서
`gst-plugin=disabled`가 맞다.

main PC의 timestamp 카메라도 Ubuntu 버전에 맞는 Aravis 0.8 설치가 필요하다.
Ubuntu 20.04 main PC라면 위의 0.8.20 소스 빌드 절차를 동일하게 사용한다.
timestamp 경로 자체는 GStreamer를 사용하지 않지만, capture PC와 동일한 설치
절차를 사용하면 머신별 환경 차이를 줄일 수 있다.

Ubuntu 버전과 설치 방식에 따라 실행 파일 이름이나 위치가 다를 수 있다.
확인은 다음처럼 한다.

```bash
command -v arv-tool-0.8 || find /usr/local/bin /usr/bin -maxdepth 1 -name 'arv-tool*'
```

Ubuntu 20.04에서 `/usr/local`에 소스 설치한 typelib은 기본 GI 검색 경로에
포함되지 않을 수 있다. 다음 환경 변수를 설정한다.

```bash
export GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0${GI_TYPELIB_PATH:+:$GI_TYPELIB_PATH}
```

Conda `(base)`의 `python3`가 아니라 시스템 Python으로 다음 명령이 성공해야
한다.

```bash
/usr/bin/python3 -c "import gi; gi.require_version('Aravis','0.8'); from gi.repository import Aravis; Aravis.update_device_list(); print(Aravis, Aravis.get_n_devices())"
```

## 2. Python 환경

### 권장: system Python venv

OS에서 설치한 `python3-gi`와 ABI가 정확히 일치하므로 이 방식이 가장
안정적이다.

```bash
cd ~/paradex
/usr/bin/python3 -m venv --system-site-packages .venv-aravis
.venv-aravis/bin/python -m pip install --upgrade pip
.venv-aravis/bin/python -m pip install -e '.[aravis]'

GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0 \
.venv-aravis/bin/python -c "import gi; gi.require_version('Aravis','0.8'); from gi.repository import Aravis; print(Aravis)"
```

### 기존 Conda 환경(`flir_env`/`robot`)에 연결

Ubuntu 20.04의 system Python과 대상 Conda 환경이 모두 Python 3.8일 때만
시스템 `python3-gi`를 Conda 환경에 연결할 수 있다. capture PC에서는 보통
`flir_env`, main PC의 timestamp/web 앱에서는 보통 `robot`을 사용한다. 먼저
반드시 버전을 비교한다.

```bash
/usr/bin/python3 --version
conda activate flir_env
python --version

# main PC라면 대신:
conda activate robot
python --version
```

둘 다 `Python 3.8.x`일 때 다음을 한 번 실행한다. `GI_TYPELIB_PATH`만 Conda
환경 변수로 저장하고, Ubuntu의 `python3-gi` 경로는 Conda site-packages의
`.pth` 파일로 **뒤에** 추가한다.

`PYTHONPATH=/usr/lib/python3/dist-packages`를 Conda 환경 변수로 지정하면
Ubuntu 20.04의 오래된 `typing_extensions` 같은 패키지가 Conda 패키지를
덮어써 FastAPI/Pydantic import가 깨지므로 사용하지 않는다.

```bash
conda activate flir_env

conda env config vars unset PYTHONPATH
conda env config vars set \
  GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0

CONDA_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
printf '%s\n' /usr/lib/python3/dist-packages > "$CONDA_SITE/ubuntu-python3-gi.pth"

conda deactivate
conda activate flir_env

conda env config vars list
python -m pip install -e '.[aravis]'
python -c "import gi; gi.require_version('Aravis','0.8'); from gi.repository import Aravis; print(Aravis)"
python -c "from typing_extensions import Self; import fastapi; print(fastapi.__version__)"
```

main PC `robot` 환경에는 위 명령의 `conda activate flir_env`만
`conda activate robot`으로 바꿔서 동일하게 적용한다.

#### 이전에 사용한 `PYTHONPATH` 방식과 복구

초기 설치에서는 다음처럼 두 경로를 모두 Conda 환경 변수로 저장했다.

```bash
conda env config vars set \
  GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0 \
  PYTHONPATH=/usr/lib/python3/dist-packages
```

이 방식은 Aravis import 자체는 간단히 해결하지만, Ubuntu 20.04의 오래된
`typing_extensions` 등이 Conda 패키지보다 먼저 로드되어 FastAPI/Pydantic을
깨뜨릴 수 있다. 기록 목적으로 남겨두되 새 설치에는 사용하지 않는다. 이미
적용했다면 `PYTHONPATH`만 해제하고 위 `.pth` 방식을 적용한다.

```bash
# 잘못된 PYTHONPATH가 conda 명령 자체의 requests/six import도 깨뜨리므로
# 현재 shell에서 먼저 임시 해제한다.
unset PYTHONPATH

# main PC의 robot 환경 복구
conda env config vars unset -n robot PYTHONPATH
conda env config vars set -n robot \
  GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0

# capture PC는 위 명령의 "-n robot"을 "-n flir_env"로 바꾼다.

conda activate robot  # capture PC라면 flir_env

CONDA_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
printf '%s\n' /usr/lib/python3/dist-packages > "$CONDA_SITE/ubuntu-python3-gi.pth"

conda deactivate
conda activate robot  # 또는 flir_env
```

Conda Python이 3.8이 아니거나 위 import가 실패하면 시스템 GI 바이너리를
강제로 섞지 말고 `.venv-aravis`를 사용한다. 설정을 되돌릴 때는 다음을
실행하고 환경을 다시 활성화한다.

```bash
conda activate flir_env
CONDA_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
rm -f "$CONDA_SITE/ubuntu-python3-gi.pth"
conda env config vars unset GI_TYPELIB_PATH
conda deactivate
conda activate flir_env
```

## 3. 카메라 NIC 설정

`192.168.*`일 필요는 없다. 현재 장비의 다음 구성을 그대로 사용할 수 있다.

```text
11.0.1.1/24
11.0.2.1/24
11.0.3.1/24
11.0.4.1/24
```

각 물리 카메라 NIC에는 서로 다른 `/24` subnet을 배정한다. 예시 netplan:

```yaml
network:
  version: 2
  ethernets:
    enp5s0:
      addresses: [11.0.1.1/24]
      mtu: 9216
    enp6s0:
      addresses: [11.0.2.1/24]
      mtu: 9216
```

```bash
sudo netplan generate
sudo netplan apply
ip -br -4 addr
```

Jumbo frame을 쓸 때는 PC NIC, 중간 switch, 카메라가 모두 지원해야 한다.

```bash
sudo ip link set dev enp5s0 mtu 9216
ip link show enp5s0
```

전체 경로가 jumbo frame을 지원하면 기본값 `9000`을 사용한다. 지원하지 않으면
NIC MTU를 1500으로 두고 다음처럼 카메라 packet size를 낮춘다.

```bash
export PARADEX_GIGE_PACKET_SIZE=1400
```

카메라 NIC를 명시하면 관리 NIC나 Docker/VPN NIC를 잘못 선택하는 것을 막을
수 있다.

```bash
export PARADEX_CAMERA_NICS=enp5s0,enp6s0,enp7s0,enp8s0
export PARADEX_GIGE_PACKET_SIZE=9000
```

main PC에 timestamp 카메라가 연결된 전용 NIC도 동일하게 IP와 MTU를 설정한다.
main PC timestamp backend는 ForceIP를 수행하지 않으므로 카메라가 해당 NIC의
subnet에 있고 Aravis discovery에 보여야 한다.

## 4. 수신 버퍼 설정

모든 GigE 카메라 연결 PC에서:

```bash
sudo tee /etc/sysctl.d/99-gige-vision.conf >/dev/null <<'EOF'
net.core.rmem_max = 16777216
net.core.rmem_default = 16777216
EOF
sudo sysctl -p /etc/sysctl.d/99-gige-vision.conf
```

NIC가 지원하는 RX ring 최대값을 확인해서 적용한다.

```bash
ethtool -g enp5s0
sudo ethtool -G enp5s0 rx <표시된 최대값>
```

## 5. 설정 파일

main PC의 [`system/current/network.json`](../../system/current/network.json):

```json
"timestamp": {
  "param": {
    "cam_type": "aravis",
    "name": "22684253"
  }
}
```

`name`은 main PC에 연결된 timestamp 카메라 serial이다. Gain/exposure는
[`system/current/camera.json`](../../system/current/camera.json)의 같은 serial
설정을 사용하며, 없으면 기본값을 사용한다.

capture PC에 배정된 serial은 기존
[`system/current/pc.json`](../../system/current/pc.json)을 그대로 사용한다.

## 6. 실행

각 capture PC에서 먼저 실행한다. Aravis/GStreamer가 기본 backend이므로
`--backend` 옵션은 생략할 수 있다.

```bash
cd ~/paradex
conda activate flir_env
PARADEX_CAMERA_NICS=enp5s0,enp6s0,enp7s0,enp8s0 \
PARADEX_GIGE_PACKET_SIZE=9000 \
GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0 \
python src/camera/server_daemon.py
```

daemon 시작 로그에서 배정된 모든 serial이 `PREPARED` 상태인지 확인한다.
그 다음 main PC에서 기존 capture 명령을 실행한다.

```bash
python src/dataset_acquisition/hri/capture_hand.py --name test
```

별도의 timestamp daemon은 필요 없다. `CaptureSession`이
`system/current/network.json`을 읽어 main PC의 Aravis timestamp 카메라를
직접 연결한다.

문제 발생 시에만 이전 backend를 명시적으로 선택할 수 있다.

```bash
python src/camera/server_daemon.py --backend pyspin
```

## 7. 검증

1. 모든 capture PC daemon을 실행한다.
2. main PC에서도 timestamp 카메라가 Aravis discovery에 표시되는지 확인한다.
3. `c`로 캡처를 시작하고 몇 초 뒤 `s`로 종료한다.
4. `raw/videos/<serial>.avi` 파일을 `ffprobe`로 확인한다.
5. main PC 결과의 `raw/timestamps/frame_id.npy`와 `timestamp.npy` 길이가
   같고 0보다 큰지 확인한다.
6. `c -> s -> c -> s`를 반복해 daemon 재시작 없이 두 번째 세션도 정상인지
   확인한다.

```bash
ffprobe <capture-PC-output>/videos/<serial>.avi
python -c "import numpy as np; print(len(np.load('<session>/raw/timestamps/frame_id.npy')), len(np.load('<session>/raw/timestamps/timestamp.npy')))"
```

## 주요 환경 변수

| 변수 | 기본값 | 설명 |
| --- | ---: | --- |
| `PARADEX_CAMERA_NICS` | 자동 탐색 | capture PC의 카메라 NIC 목록 |
| `PARADEX_GIGE_PACKET_SIZE` | `9000` | Jumbo frame 불가 시 `1400` |
| `PARADEX_GIGE_HEARTBEAT_MS` | `10000` | GigE control heartbeat |
| `PARADEX_ARAVIS_BUFFERS` | `64` | capture 카메라 재사용 버퍼 수 |
| `PARADEX_STREAM_POLL_TIMEOUT_US` | `200000` | Aravis stream poll 간격 |
| `PARADEX_FIRST_FRAME_TIMEOUT` | `10.0` | UTG ON 뒤 첫 프레임 검증 시간 |
| `PARADEX_JPEG_QUALITY` | `95` | AVI MJPEG 품질 |
| `PARADEX_PREVIEW_WIDTH` | `640` | main PC 웹 preview JPEG 폭 |
| `PARADEX_PREVIEW_FPS` | `5` | preview branch 최대 FPS |
| `PARADEX_PREVIEW_JPEG_QUALITY` | `70` | preview JPEG 품질 |

## Main PC 웹 캡처 테스트

capture PC에는 브라우저 UI가 없으며 `server_daemon.py`가 포트 `5484`에서
최신 preview JPEG만 제공한다. 실제 녹화 branch와 preview branch는 GStreamer
`tee` 뒤에서 분리되고 preview queue는 leaky이므로 main PC 또는 브라우저가
느려져도 AVI 녹화를 막지 않는다.

main PC 웹 의존성을 설치한다.

```bash
cd ~/paradex
python -m pip install -e '.[site]'
```

각 capture PC에서 기존 daemon을 먼저 실행한 뒤 main PC에서 다음을 실행한다.

```bash
python -m paradex.dataset_acqusition.capture_site --host 0.0.0.0 --port 8000
```

브라우저에서는 main PC의 `http://<main-pc-ip>:8000`만 연다. main PC가
`system/current/pc.json`의 serial/IP 매핑을 사용해 각 capture PC의 JPEG를
proxy하므로 capture PC 웹페이지에 직접 접속할 필요가 없다.

테스트 사이트의 저장 경로는 다음과 같다.

```text
~/shared_data/capture/site_test/<dataset>/<episode>/
```

내부 `raw/timestamps`, 카메라 파라미터 및 capture PC의 AVI 배치는 기존
`CaptureSession`/Paradex 저장 형식을 그대로 따른다. 현재 preview는 녹화
세션 중에만 갱신되며 idle 상태에서는 마지막 이미지가 없거나 404가 정상이다.

필요한 방화벽 포트:

- capture PC: TCP `5480`~`5484`
- main PC 웹: TCP `8000`

## 문제 해결

- `Aravis is unavailable`: 실행 중인 Python이 OS의 GI 패키지를 보지 못한다.
  `--system-site-packages` venv 또는 OS Python을 사용한다.
- `Device not found`: serial 문자열을 직접 device ID로 사용하지 않는다. 코드는
  discovery 결과에서 serial에 대응하는 정확한 Aravis device ID를 찾는다.
- 프레임 0개: UTG 배선, `Line0`, RisingEdge, 카메라 subnet을 확인한다.
- `not-negotiated`: capture PC의 GStreamer plugin 설치와 Bayer format/해상도를
  확인한다.
- 한 카메라만 저장됨: NIC RX ring, receive buffer, jumbo MTU 일치 여부와 디스크
  쓰기 속도를 확인한다.
