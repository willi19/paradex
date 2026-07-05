# 카메라 시스템

Paradex **카메라 서브시스템**이 실제로 어떻게 돌아가는지 그림과 함께 훑는 문서입니다.
코드를 열기 전에 이걸 먼저 읽으면 각 파일이 왜 그렇게 생겼는지 감이 잡힙니다. 카메라는
전체 리그의 한 부분일 뿐이고, robot·capture·pipeline 가이드도 앞으로 "Guide" 아래 나란히
들어올 수 있습니다.

- 자세한 재설계 제안: [design/camera-recording-redesign.md](https://github.com/willi19/paradex/blob/main/design/camera-recording-redesign.md)
- 함수·클래스 단위 레퍼런스: {doc}`API Reference <autoapi/index>`

## 왜 이렇게 나눠져 있나

카메라 한 대가 기가비트 랜을 거의 다 먹습니다. 그래서 수십 대를 한 컴퓨터에 붙일 수 없어,
**캡처 PC 6대**에 나눠 답니다. 대신 **메인 PC**는 카메라를 직접 만지지 않습니다 — 각 캡처
PC에서 도는 데몬에게 "찍어라 / 멈춰라" 명령만 보냅니다.

여러 대에 나눠 달면 자연히 이런 걱정이 생기죠. *"그럼 카메라마다 셔터가 제각각 열리는 거
아냐?"* 그래서 모든 카메라를 **하나의 하드웨어 트리거(UTGE900)** 에 물립니다. 트리거 펄스가
전기신호로 모든 카메라에 동시에 도달하니, 어느 PC에 붙어 있든 정확히 같은 순간에 찍힙니다.

```{mermaid}
flowchart LR
    subgraph Main["메인 PC"]
      ORCH["capture / inference<br/>스크립트"]
      RCC["remote_camera_controller"]
      ORCH --> RCC
    end
    subgraph Cap["캡처 PC (×6)"]
      D["server_daemon"]
      CL["CameraLoader"]
      CAM["Camera × k (PySpin)"]
      D --> CL --> CAM
    end
    GEN["UTGE900<br/>신호 발생기"]
    RCC -- "ZMQ: register / start / stop / heartbeat" --> D
    GEN -. "하드웨어 트리거 (전기신호)" .-> CAM
```

## 메인 PC가 캡처 PC와 대화하는 법

메인 PC 쪽 `remote_camera_controller`는 각 데몬에 붙어 **컨트롤러 락**을 하나 잡습니다. 두
사람이 동시에 같은 카메라를 건드리지 못하게 하려는 겁니다. 락을 잡으면 모드를 시작시키고,
그 뒤로는 0.1초마다 **heartbeat**를 보내 "나 아직 살아있다"를 알립니다.

만약 15초 동안 heartbeat가 끊기면 데몬은 스스로 락을 풀고 카메라를 멈춥니다 — 컨트롤러가
죽었을 때 카메라가 계속 돌아가는 걸 막는 안전장치입니다. (다만 이 타임아웃이 뒤에 나오는
hang 문제와 얽히니, 그건 아래에서 다시 봅니다.)

```{mermaid}
sequenceDiagram
    participant M as 메인 PC (컨트롤러)
    participant D as 캡처 PC (데몬)
    M->>D: register (락 잡기)
    M->>D: start(mode, sync, save_path, fps, exposure, gain)
    D->>D: CameraLoader.start → 카메라별 acquisition 시작
    loop 약 0.1초마다
      M->>D: heartbeat
      D-->>M: ok / 카메라 에러
    end
    M->>D: stop
    M->>D: end (락 반환)
```

## 카메라 한 대 안에서 벌어지는 일 (가장 중요)

여기가 핵심 멘탈 모델입니다. 카메라마다 **캡처 스레드가 하나** 돌면서 프레임을 쉬지 않고
잡아옵니다. 잡은 프레임을 *어디로 보내느냐* 는 mode가 정하는데, 이 "보내는 곳"을 sink라고
부릅니다.

- **stream** — 공유 메모리(SHM)에 얹습니다. 실시간으로 프레임이 필요한 소비자(예: 6D 포즈
  초기화)가 `MultiCameraReader`로 바로 읽어갑니다.
- **video** / **full** — 디스크에 `.avi`로 녹화합니다 (`full`은 녹화 + SHM을 동시에).
- **image** — 한 장만 저장합니다.

즉 "카메라가 프레임을 만든다"와 "그 프레임을 어디에 쓴다"는 원래 별개의 축인데, 지금은
mode 하나가 이 둘을 같이 묶고 있습니다. 이 점을 기억해 두세요.

```{mermaid}
flowchart TD
    T["캡처 스레드<br/>continuous_acquire()"] --> G["get_image()<br/>GetNextImage(GRAB_TIMEOUT_MS)"]
    G -->|프레임| SHM["SHM 더블버퍼<br/>(stream / full)"]
    G -->|프레임| VID["cv2.VideoWriter .avi<br/>(video / full)"]
    G -.->|프레임 없음 → (None, None)| T
    SHM --> R["MultiCameraReader → 소비자"]
    VID --> DISK["캡처 디스크"]
```

> 그래서 stream을 켜둔 채 녹화를 시작하려면, 지금 구조에선 acquisition을 통째로 멈췄다
> 다시 켜야 합니다. [재설계](https://github.com/willi19/paradex/blob/main/design/camera-recording-redesign.md)는
> 이 둘을 떼어내서, 녹화를 껐다 켜는 걸 재시작 없이 하자고 제안합니다.

## "싱크가 맞는다"는 게 무슨 뜻

트리거 펄스가 한 번 칠 때마다 **모든 카메라의 frame_id가 동시에 1씩 올라갑니다.** 그래서
어느 순간을 집어봐도 카메라들이 같은 번호를 들고 있어야 정상입니다.

`sync_check.py`는 이걸 카메라들 frame_id의 최대–최소 차이(spread)로 확인합니다. 차이가 계속
0~1이면 잘 맞는 것이고, 한 카메라만 자꾸 뒤처져 차이가 벌어지면 그 카메라가 프레임을
흘리거나 트리거를 못 받고 있다는 신호입니다.

```{mermaid}
flowchart LR
    GEN["트리거 펄스 #N"] --> C1["카메라 A → 프레임 #N"]
    GEN --> C2["카메라 B → 프레임 #N"]
    GEN --> C3["카메라 C → 프레임 #N"]
    C1 & C2 & C3 --> CHK["spread = 최대-최소 id<br/>0~1 이면 싱크 정상"]
```

## 카메라가 멈췄던 이유, 그리고 고친 것

최근에 잡은 실제 버그입니다. 랜선이 빠지거나 트리거가 멈춰서 **프레임이 안 들어오면**, 예전
`get_image()`는 프레임을 **무한정 기다렸습니다.** 캡처 스레드가 거기서 멈춰버리니 stop 신호를
확인하지 못하고, 그 결과 `Camera.stop()`도 영원히 대기하면서 **데몬 전체가 굳어버립니다.** 이
상태가 되면 `pkill`로 메인 스크립트를 죽여도, 캡처 PC의 데몬은 굳은 채 살아있어서 다음
실행이 카메라를 못 켰습니다.

고친 방법은 단순합니다. `get_image()`가 무한 대기 대신 **1초 timeout**을 걸고, 그 안에
프레임이 안 오면 `(None, None)`을 돌려줍니다. 그러면 루프가 다시 stop/exit 신호를 확인할 수
있어, 프레임이 끊겨도 stop이 정상적으로 반환됩니다.

```{mermaid}
flowchart TD
    subgraph Before["예전 (버그)"]
      A["랜선 빠짐 / 트리거 꺼짐"] --> B["GetNextImage() 무한 대기"]
      B --> C["루프가 stop을 다시 못 봄"]
      C --> E["Camera.stop() 영원히 대기 → 데몬 굳음"]
    end
    subgraph After["지금 (P4 수정)"]
      A2["랜선 빠짐 / 트리거 꺼짐"] --> B2["GetNextImage(1000ms) → (None, None)"]
      B2 --> C2["루프가 start/exit 재확인"]
      C2 --> E2["stop()/end() 정상 반환 (유한 대기)"]
    end
```

혹시 카메라가 하드웨어 레벨에서 굳어버렸다면, 메인 PC에서 `python src/camera/reset_cameras.py`로
데몬을 강제 종료한 뒤 재기동하면 됩니다. 이 수정이 실제로 먹히는지는
`src/validate/camera_system/hang_recovery.py`로 검증합니다.

## 게인·노출은 어디서 오나

카메라별 기본값은 `system/current/camera.json`에 들어 있습니다. 실제로 어떤 값을 쓸지는
**명시적으로 넘긴 값 > camera.json의 카메라별 값 > 기본값** 순서로, 항상 이 순서대로
결정됩니다. 그래서 노출 스윕처럼 한 번 값을 바꿔 써도, 그 값이 다음 캡처로 슬그머니 새어
들어가지 않습니다. 화면을 보면서 실시간으로 맞추려면 `src/util/camera_tuning/live_tuner.py`를
쓰세요.

```{mermaid}
flowchart LR
    A["start(exposure=None, gain=None)"] --> B{"명시 인자 있음?"}
    B -->|예| U["그 값 사용"]
    B -->|아니오| C["camera.json[serial]"]
    C -->|없음| D["기본값 2500us / 3dB"]
```

## 코드로 들어가기: 층위

지금까지가 "어떻게 돌아가나"였다면, 여기부터는 **코드가 실제로 어떻게 짜여 있나**입니다.
명령은 위에서 아래로 흐르고, 각 층이 이런 일을 합니다.

| 층 | 클래스 / 파일 | 하는 일 |
|---|---|---|
| 메인 PC | `remote_camera_controller` | 각 캡처 PC에 명령 전송, heartbeat 유지 |
| 캡처 PC 데몬 | `camera_server_daemon` | 명령 수신 → CameraLoader 호출, 락 관리 |
| 묶음 | `CameraLoader` | 카메라 N대를 스레드로 동시에 start/stop |
| 카메라 1대 | `Camera` | 이벤트로 캡처 스레드 조종, sink 분배 |
| 하드웨어 | `PyspinCamera` | 실제 PySpin 호출 (grab, 트리거, 노출) |

## `Camera` — 이벤트로 스레드를 조종한다

`Camera` 한 개는 생성될 때 **캡처 스레드 하나**(`run`)를 띄우고, 그 스레드와 바깥 호출자는
**Event 여러 개**로 신호를 주고받습니다. 이게 이 코드에서 제일 안 보이면서 제일 중요한
부분입니다.

```python
self.event = {
    "start": Event(), "stop": Event(), "exit": Event(),
    "acquisition": Event(), "connection": Event(), "error": Event(), ...
}
```

캡처 스레드는 이 루프를 돕니다 — `start`가 켜지면 잡기 시작하고, `exit`가 켜지면 빠져나옵니다:

```python
def run(self):
    self.connect_camera()                       # PySpin 카메라 열기
    while not self.event["exit"].is_set():
        if self.event["start"].is_set():
            if self.mode in ["full", "video", "stream"]:
                self.continuous_acquire()       # 연속 캡처 루프
            else:
                self.single_acquire()           # 한 장
        time.sleep(0.001)
    self.release()
```

바깥에서 `start()`를 부르면 단순히 플래그만 세우는 게 아니라, **캡처 스레드가 실제로 시작할
때까지 기다립니다**(악수):

```python
def start(self, mode, syncMode, ...):
    ...
    self.event["stop"].clear()
    self.event["start"].set()          # "캡처 스레드야, 시작해"
    self.event["acquisition"].wait()   # 진짜 시작될 때까지 대기
```

`continuous_acquire`가 캡처 루프의 몸통입니다. 프레임을 잡아 sink로 나눕니다:

```python
self.camera.start("continuous", self.syncMode, self.fps, ...)  # BeginAcquisition
self.event["acquisition"].set()                                # 위 start()의 대기를 풀어줌
while self.event["start"].is_set() and not self.event["exit"].is_set():
    frame, frame_data = self.camera.get_image()
    if frame is None:
        continue                       # 프레임 없음(timeout) → while 조건 재확인
    if save_video: video_writer.write(frame)     # video / full sink
    if stream:     ...                            # SHM 더블버퍼에 씀 (write_flag 토글)
self.camera.stop()
self.event["stop"].set()               # 아래 stop()의 대기를 풀어줌
```

**멈추는 것도 이벤트로 합니다.** `stop()`은 `event["start"]`를 끄고 `event["stop"]`을 기다립니다.
위 루프의 `while` 조건이 다음 바퀴에서 False가 되어 빠져나오고, 스레드가 `event["stop"]`을
켜주면 `stop()`이 반환됩니다.

```python
def stop(self, timeout=5.0):
    self.event["start"].clear()
    self.event["stop"].wait(timeout=timeout)   # 유한 대기 (P4)
```

> 앞의 hang 버그가 정확히 여기였습니다: `get_image()`가 무한 대기하면 루프가 다음 바퀴로 못
> 가서 `while` 조건을 영영 재확인 못 하고, 그래서 `event["stop"]`도 안 켜져서 `stop()`이 영영
> 반환 안 됐던 거죠. 그래서 고친 곳도 `get_image()` 한 군데입니다.

## `PyspinCamera` — 실제 하드웨어

`Camera`가 `self.camera`로 들고 있는 게 `PyspinCamera`입니다. 두 메서드가 핵심입니다.

```python
def get_image(self):
    try:
        pImageRaw = self.cam.GetNextImage(GRAB_TIMEOUT_MS)  # 유한 timeout (P4)
    except ps.SpinnakerException:
        return None, None                                   # 프레임 없음
    ...
    return frame, frame_data
```

`start`는 **바뀐 설정만** 다시 적용하고 `BeginAcquisition`을 부릅니다 (매번 전부 재설정하면
느리고 sync가 흔들리니까):

```python
if syncMode:                       self._configureTrigger()
if gain     != self.gain:          self._configureGain()
if exposure != self.exposure_time: self._configureExposure()
self.cam.BeginAcquisition()
```

## `CameraLoader` — 여러 대를 동시에

카메라 N대를 **스레드로 동시에** 시작합니다 (순차로 하면 느리니까). 앞서 말한 게인/노출 결정
로직도 여기 있습니다:

```python
for camera, path in zip(self.cameralist, save_paths):
    cfg = self.cam_config.get(camera.name, {})
    e = exposure_time if exposure_time is not None else cfg.get("exposure", DEFAULT_EXPOSURE)
    g = gain          if gain          is not None else cfg.get("gain", DEFAULT_GAIN)
    Thread(target=camera.start, args=(mode, syncMode, path, fps, e, g)).start()
```

`stop`/`end`도 같은 팬아웃을 재사용합니다 (`_broadcast`).

## `camera_server_daemon` — 명령 루프 + 락

데몬은 ZMQ로 명령을 받아 `execute_command`로 분기합니다. 명령 소켓에 **15초 타임아웃**이
걸려 있어서, 컨트롤러가 그 안에 아무 명령(heartbeat 포함)도 안 보내면 자동으로 락을 풀고
카메라를 멈춥니다:

```python
self.command_socket.setsockopt(zmq.RCVTIMEO, 15000)   # 15초
while True:
    try:
        cmd = self.command_socket.recv_json()
        resp = self.execute_command(cmd)              # register / start / stop / heartbeat / ...
        self.command_socket.send_json(resp)
    except zmq.Again:                                 # 15초 무명령 → 자동 해제
        ... 컨트롤러 락 반환 + 카메라 정지 ...
```

`register`가 락을 잡고, 다른 컨트롤러가 이미 잡고 있으면 명령을 거부합니다.

## `remote_camera_controller` — 메인 PC 쪽 루프

메인 PC는 백그라운드 스레드에서 루프를 돌며, 평소엔 heartbeat를 보내고 `start`/`stop`
이벤트가 서면 그 명령을 **모든 캡처 PC에 병렬로** 보냅니다:

```python
while not self.exit_event.is_set():
    cmd = {'action': 'heartbeat'}
    if self.start_event.is_set(): cmd = {'action': 'start', 'mode': ..., ...}
    if self.stop_event.is_set():  cmd = {'action': 'stop'}
    response = self.send_command(cmd)     # PC마다 스레드 하나로 동시 전송
    time.sleep(0.1)
```

즉 사용자가 부르는 `rcc.start(...)`는 이벤트만 세우고, 실제 전송은 이 루프가 합니다. — 이래서
`Camera`(이벤트로 스레드 조종)와 `remote_camera_controller`(이벤트로 루프 조종)가 같은 패턴을
씁니다.

## 어디를 보면 되나

| 이걸 하고 싶으면… | 파일 |
|-----------------|------|
| 메인 PC에서 카메라 제어 | `paradex/io/camera_system/remote_camera_controller.py` |
| 캡처 PC 데몬 이해 | `paradex/io/camera_system/camera_server_daemon.py` |
| acquisition 루프 & sink | `paradex/io/camera_system/camera.py` (`continuous_acquire`) |
| 저수준 PySpin (grab, 설정, 트리거) | `paradex/io/camera_system/pyspin.py` |
| 멀티카메라 값 해석 | `paradex/io/camera_system/camera_loader.py` |
| hang 수정 / 싱크 검증 | `src/validate/camera_system/{hang_recovery,sync_check}.py` |
| 게인·노출 실시간 튜닝 | `src/util/camera_tuning/live_tuner.py` |
| 굳은 카메라 복구 | `src/camera/reset_cameras.py` |
