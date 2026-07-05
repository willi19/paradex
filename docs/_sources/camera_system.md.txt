# Camera System

Paradex 카메라 서브시스템 개발 문서. 분산 멀티카메라 캡처의 구조, 컴포넌트, 상태 흐름, 설정,
에러 처리를 다룬다. 코드를 읽기 전에 이 문서로 전체 구조를 잡을 것.

- 재설계 제안: [design/camera-recording-redesign.md](https://github.com/willi19/paradex/blob/main/design/camera-recording-redesign.md)
- 심볼 단위 API: {doc}`API Reference <autoapi/index>`

---

## 1. Overview

카메라 수십 대가 **캡처 PC 6대**에 분산되어 있고(카메라 1대가 기가비트 랜을 거의 다 쓰므로),
**메인 PC**는 카메라를 직접 만지지 않고 각 캡처 PC의 데몬에 명령을 보내 제어한다. 모든 카메라는
하나의 **하드웨어 트리거(UTGE900)** 에 물려 동기 촬영된다.

스택은 5개 층으로 구성된다. 명령은 위에서 아래로 흐른다.

```{mermaid}
flowchart LR
    subgraph Main["메인 PC"]
      ORCH["capture / inference"]
      RCC["remote_camera_controller"]
      ORCH --> RCC
    end
    subgraph Cap["캡처 PC (×6)"]
      D["server_daemon"]
      CL["CameraLoader"]
      CAM["Camera × k → PyspinCamera"]
      D --> CL --> CAM
    end
    GEN["UTGE900"]
    RCC -- "ZMQ: register/start/stop/heartbeat" --> D
    GEN -. "하드웨어 트리거" .-> CAM
```

| 층 | 컴포넌트 | 위치 | 책임 |
|----|----------|------|------|
| Control | `remote_camera_controller` | 메인 PC | 명령 병렬 전송, heartbeat |
| Service | `camera_server_daemon` | 캡처 PC | 명령 수신·분기, 컨트롤러 락 |
| Group | `CameraLoader` | 캡처 PC | 카메라 N대 묶음 제어, 파라미터 해석 |
| Device | `Camera` | 캡처 PC | 캡처 스레드 상태머신, sink 분배 |
| Driver | `PyspinCamera` | 캡처 PC | PySpin SDK 호출 |

---

## 2. Core Concepts

| 용어 | 의미 |
|------|------|
| **Mode** | `image` / `video` / `stream` / `full`. 캡처 방식과 저장 위치를 함께 지정. |
| **Sink** | 프레임의 목적지. `stream`→SHM, `video`→`.avi`, `image`→파일 1장, `full`→SHM+`.avi`. |
| **Acquisition** | 카메라당 캡처 스레드가 프레임을 연속으로 잡는 것. sink와는 별개의 축. |
| **Hardware sync** | 트리거 펄스마다 전 카메라의 `frame_id`가 동시에 1 증가. 같은 순간 = 같은 id. |
| **Controller lock** | 데몬당 컨트롤러 1개만 허용. `register`로 획득, 15초 무-heartbeat 시 자동 해제. |

---

## 3. Components

### 3.1 `remote_camera_controller` (메인 PC)

- **책임**: 캡처 PC들에 명령을 병렬 전송하고 heartbeat로 연결을 유지.
- **주요 메서드**: `initialize()`(핑 확인·소켓·락), `start(mode, sync, ...)`, `stop()`, `end()`, `run()`(백그라운드 루프).
- **인터페이스**: ZMQ REQ → 각 데몬의 `command_port(5482)`, 상태 확인은 `ping_port(5480)`.
- **주의**: `start()`/`stop()`은 **이벤트만 세우고**, 실제 명령 전송은 `run()` 루프가 수행한다.

```python
def run(self):
    self.initialize()
    while not self.exit_event.is_set():
        cmd = {'action': 'heartbeat'}
        if self.start_event.is_set(): cmd = {'action': 'start', 'mode': ..., ...}
        if self.stop_event.is_set():  cmd = {'action': 'stop'}
        response = self.send_command(cmd)     # PC마다 스레드 1개로 동시 전송
        time.sleep(0.1)
```

### 3.2 `camera_server_daemon` (캡처 PC)

- **책임**: 명령 수신 → `execute_command`로 분기 → `CameraLoader` 호출. 단일 컨트롤러 락 관리.
- **포트**: `ping 5480`(REP), `monitor 5481`(PUB 상태 방송), `command 5482`(REP).
- **명령**: `register` / `start` / `stop` / `heartbeat` / `reload` / `end`.
- **락 + 타임아웃**: command 소켓에 `RCVTIMEO 15s`. 15초간 무-명령이면 락을 풀고 카메라를 멈춘다(컨트롤러 사망 대비).

```python
self.command_socket.setsockopt(zmq.RCVTIMEO, 15000)   # 15초
while True:
    try:
        cmd = self.command_socket.recv_json()
        resp = self.execute_command(cmd)              # register/start/stop/heartbeat/...
        self.command_socket.send_json(resp)
    except zmq.Again:                                 # 15초 무-명령 → 락 해제 + 카메라 정지
        ...
```

### 3.3 `CameraLoader` (캡처 PC)

- **책임**: 카메라 N대를 스레드로 동시에 start/stop. 게인/노출을 **카메라별로** 해석.
- **해석 규칙**: `명시 인자 > camera.json[serial] > 기본값`. `None`은 "camera.json 사용"이지 "직전 값 유지"가 아니다(§7).

```python
for camera, path in zip(self.cameralist, save_paths):
    cfg = self.cam_config.get(camera.name, {})
    e = exposure_time if exposure_time is not None else cfg.get("exposure", DEFAULT_EXPOSURE)
    g = gain          if gain          is not None else cfg.get("gain", DEFAULT_GAIN)
    Thread(target=camera.start, args=(mode, syncMode, path, fps, e, g)).start()
```

`stop`/`end`는 동일한 팬아웃 헬퍼(`_broadcast`)를 재사용한다.

### 3.4 `Camera` (캡처 PC) — 상태 머신

- **스레딩 모델**: 생성 시 캡처 스레드(`run`) 1개를 띄운다. 바깥 호출자와 캡처 스레드는 **Event 집합**으로 동기화한다(플래그가 아니라 악수).
- **상태** (`get_state()` 기준):

```{mermaid}
stateDiagram-v2
    [*] --> CONNECTING
    CONNECTING --> READY: connect_camera()
    READY --> STARTING: start()
    STARTING --> CAPTURING: acquisition set
    CAPTURING --> READY: stop()
    CAPTURING --> ERROR: exception
    ERROR --> READY: error_reset()
    READY --> STOPPED: end()
    STOPPED --> [*]
```

- **핵심 루프**: `continuous_acquire`(stream/video/full), `single_acquire`(image).

```python
# run(): start가 켜지면 캡처, exit면 종료
while not self.event["exit"].is_set():
    if self.event["start"].is_set():
        self.continuous_acquire() if self.mode in ["full","video","stream"] else self.single_acquire()
    time.sleep(0.001)

# start(): 이벤트를 세우고 캡처 스레드가 실제 시작할 때까지 대기(악수)
self.event["start"].set()
self.event["acquisition"].wait()

# continuous_acquire(): 몸통
self.camera.start("continuous", self.syncMode, self.fps, ...)   # BeginAcquisition
self.event["acquisition"].set()
while self.event["start"].is_set() and not self.event["exit"].is_set():
    frame, frame_data = self.camera.get_image()
    if frame is None: continue          # timeout → while 조건 재확인
    if save_video: video_writer.write(frame)
    if stream:     ...                  # SHM 더블버퍼(write_flag 토글)
self.camera.stop(); self.event["stop"].set()
```

### 3.5 `PyspinCamera` (캡처 PC) — 드라이버

- **책임**: PySpin SDK 직접 호출. `get_image()`(grab), `start()`(설정+`BeginAcquisition`), `_configure*`(gain/exposure/trigger/framerate).

```python
def get_image(self):
    try:
        pImageRaw = self.cam.GetNextImage(GRAB_TIMEOUT_MS)   # 유한 timeout (§8)
    except ps.SpinnakerException:
        return None, None                                    # 프레임 없음
    ...
    return frame, frame_data

def start(self, mode, syncMode, frame_rate=None, gain=None, exposure_time=None):
    if syncMode:                       self._configureTrigger()   # 바뀐 것만 재적용
    if gain     != self.gain:          self._configureGain()
    if exposure != self.exposure_time: self._configureExposure()
    self.cam.BeginAcquisition()
```

---

## 4. Command Flow

```{mermaid}
sequenceDiagram
    participant M as 메인 PC (controller)
    participant D as 캡처 PC (daemon)
    M->>D: register (락 획득)
    M->>D: start(mode, sync, save_path, fps, exposure, gain)
    D->>D: CameraLoader.start → 카메라별 acquisition
    loop 약 0.1s 마다
      M->>D: heartbeat
      D-->>M: ok / 카메라 에러
    end
    M->>D: stop
    M->>D: end (락 반환)
```

---

## 5. Data Path: Acquisition → Sinks

카메라당 캡처 스레드 1개가 프레임을 잡아, mode에 따라 sink로 분배한다. "프레임을 만든다"와
"어디로 보낸다"는 별개 축인데, 현재는 mode 하나가 둘을 묶고 있다(재설계 대상).

```{mermaid}
flowchart TD
    T["continuous_acquire()"] --> G["get_image()"]
    G -->|프레임| SHM["SHM 더블버퍼 (stream/full)"]
    G -->|프레임| VID["VideoWriter .avi (video/full)"]
    G -.->|없음 → (None,None)| T
    SHM --> R["MultiCameraReader → 소비자"]
```

---

## 6. Hardware Sync

트리거 펄스마다 전 카메라의 `frame_id`가 동시에 증가하므로, 어느 순간이든 카메라들은 같은 id를
가져야 한다. `sync_check.py`가 카메라 간 `frame_id` 스프레드(최대−최소)로 검증한다: 0~1이면 정상,
지속적으로 벌어지면 프레임 드롭 또는 트리거 미수신.

```{mermaid}
flowchart LR
    GEN["트리거 #N"] --> C1["cam A #N"]
    GEN --> C2["cam B #N"]
    GEN --> C3["cam C #N"]
    C1 & C2 & C3 --> CHK["spread = max-min<br/>0~1 = OK"]
```

---

## 7. Configuration — Gain / Exposure

카메라별 기본값은 `system/current/camera.json`에 있다. 값 결정은 항상 다음 순서다.

```{mermaid}
flowchart LR
    A["start(exposure=None, gain=None)"] --> B{"명시 인자?"}
    B -->|예| U["그 값"]
    B -->|아니오| C["camera.json[serial]"]
    C -->|없음| D["기본 2500us / 3dB"]
```

`None`은 "camera.json 값 사용"이므로, 노출 스윕 같은 일회성 override가 다음 캡처로 새지 않는다.
화면을 보며 실시간 조정: `src/util/camera_tuning/live_tuner.py`.

---

## 8. Error Handling & Recovery

**프레임 유실 hang (P4).** 랜 드롭·트리거 정지로 프레임이 끊기면, 예전 `get_image()`는 무한
대기했다. 캡처 스레드가 §3.4 루프에서 멈춰 `while` 조건을 재확인하지 못하고, `event["stop"]`이
안 켜져 `stop()`이 영영 반환되지 않아 데몬 전체가 굳었다(→ `pkill` 후에도 재시작 불가).

```{mermaid}
flowchart LR
    subgraph Before["예전"]
      B1["GetNextImage() 무한 대기"] --> B2["stop() 영영 반환 안 됨 → 데몬 굳음"]
    end
    subgraph After["P4 수정"]
      A1["GetNextImage(1000ms) → (None,None)"] --> A2["루프 재확인 → stop()/end() 정상 반환"]
    end
```

**복구**: 하드웨어 레벨로 굳었으면 메인 PC에서 `python src/camera/reset_cameras.py`(데몬 `pkill -9`
후 재기동). **검증**: §9.

---

## 9. Validation

| 스크립트 | 검증 대상 | 하드웨어 |
|----------|-----------|----------|
| `src/validate/camera_system/hang_recovery.py` | 프레임 유실 시 stop/end 무-hang (watchdog로 hang=FAIL) | 필요 |
| `src/validate/camera_system/hang_recovery_mock.py` | `get_image` 유한 timeout + `(None,None)` 계약 | 불필요 |
| `src/validate/camera_system/sync_check.py` | 카메라 간 `frame_id` 정렬 | 필요 |

---

## 10. File Reference

| 대상 | 파일 |
|------|------|
| 메인 PC 제어 | `paradex/io/camera_system/remote_camera_controller.py` |
| 캡처 PC 데몬 | `paradex/io/camera_system/camera_server_daemon.py` |
| 묶음 제어·파라미터 해석 | `paradex/io/camera_system/camera_loader.py` |
| 캡처 스레드·sink | `paradex/io/camera_system/camera.py` (`continuous_acquire`) |
| PySpin 드라이버 | `paradex/io/camera_system/pyspin.py` |
| 실시간 튜너 | `src/util/camera_tuning/live_tuner.py` |
| 굳은 카메라 복구 | `src/camera/reset_cameras.py` |
