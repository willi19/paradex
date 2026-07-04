# 설계: 카메라 acquisition / recording 재설계

**상태:** 제안 (아직 코드 변경 없음)
**작성:** Claude와 함께 작성, 2026-06
**범위:** `paradex/io/camera_system/` + 카메라 mode API를 쓰는 모든 호출부 (전부 마이그레이션)
**동기가 된 파일:** `AutoDex/src/execution/run_auto.py` (현재 프로덕션 루프)

---

## 1. 요약

지금 카메라 API는 **카메라가 무엇을 하는지**(프레임 acquisition)와 **프레임이 어디로
가는지**(SHM stream / `.avi` 파일 / `.png` 스틸)를 하나의 `mode` 인자
(`stream | video | image | full`)로 뭉쳐놨다. 그래서 sink를 바꾸려면 매번 stop +
restart가 필요하고, 그때마다 **PySpin acquisition을 다시 arm** 하게 된다.
프로덕션 루프들(`run_auto.py`, `reset_test.py`, `naive_drop.py`, `reorient_drop.py`)은
trial 한 번에 mode를 3~4번 바꾸므로, 모든 카메라에 대해 trial마다 acquisition을
3~4번씩 재-arm 한다 — 이게 지금 겪는 flakiness, sync 깨짐, 반복되는 에러의 주범이다.

**제안:** **acquisition**(세션당 한 번만 arm, 계속 돌아감)과 **sink**(live 동안 SHM은
항상 켜짐, video 파일은 런타임 토글, 스틸은 필요할 때 캡처)를 분리한다. `mode` 문자열을
명시적인 작은 API로 교체하고, 고정 상수들을 config로 빼고, recording이 wall-clock이
아니라 로봇 모션에 프레임 단위로 정렬되도록 sync trigger를 기록한다.

이 문서는 계획만 담는다. 이미 정한 것: **계획 먼저 작성**, 그리고 실제 구현 때는
**모든 호출부를 마이그레이션**한다 (장기적으로 API 이중 유지 안 함).

---

## 2. 배경 — 세 가지 문제 (프로덕션 사용에서 나온 것)

### P1. "video mode vs stream이 문제다"
`run_auto.py`는 항상 켜진 `stream`을 유지하고(SHM 프레임이 FoundPose init에 들어감),
execution을 녹화하려면 stream을 내리고 `video`를 올린 뒤, label 스틸을 위해 다시
`image`로 바꾸고, 끝나면 `stream`으로 되돌린다:

```
startup:      rcc.start("stream", …)                       # init 소비자가 SHM 읽음
step 4:       rcc.stop(); rcc.start("video", True, raw)     # execution 녹화
              rcc.stop()
step 5:       rcc.start("image", False, label); rcc.stop()  # label 스냅샷
trial end:    rcc.start("stream", …)                        # 다음 init 위해 재개
```

`start`마다 acquisition을 재-arm 하고(`continuous_acquire` → `self.camera.start("continuous", …)`,
[camera.py:237](../paradex/io/camera_system/camera.py#L237)), `stop`마다 pyspin
`stop()`을 호출한다([camera.py:312](../paradex/io/camera_system/camera.py#L312)).
하드웨어 sync 카메라에서 세션 중간에 반복 재-arm 하는 것이 바로 sync/frame-drop 에러가
터지는 지점이다.

코드 자체가 해법이 가능함을 보여준다: **`full` mode는 SHM stream과 video 저장을 동시에**
한다([camera.py:226-227](../paradex/io/camera_system/camera.py#L226-L227)). 그리고
`AutoDex/src/experiment/reset/reorient_drop.py:1118`은 이미 이걸 이용하고 있다
(`rcc.start("full", True, video_rel)  # "full" = video AVI + SHM stream`). sink들은
서로 배타적이지 않다 — API만 배타적인 척하고 있을 뿐이다.

### P2. "hard rt process"
execution 경로는 타이밍에 민감한데 조율(coordination)이 안 되어 있다:

```python
rcc.stop()                          # stream 정지
rcc.start("video", True, raw_rel)   # 5개 PC로 블로킹 fan-out, 최대 30초 timeout
timestamp_monitor.start(...)
executor.start_recording(...)
sync_generator.start(fps=30)        # 하드웨어 트리거가 녹화 시작 "이후"에 켜짐
... 로봇 모션 ...
```

- **순서가 race다.** `syncMode=True`면 `sync_generator.start` 전에는 프레임이 안 흐르지만,
  recording-start / sync-start / motion-start가 각각 별개의 Python 호출이고 그 사이
  지연이 가변적이다. 정렬은 사후 `timestamp_monitor`에 의존한다.
- **critical path에 블로킹 start가 있다.** `remote_camera_controller.send_command`는
  start/stop에 30초 timeout을 쓰고([remote_camera_controller.py:81](../paradex/io/camera_system/remote_camera_controller.py#L81))
  5개 PC 전부가 ack 해야 리턴한다 — 모션 직전에 예측 불가능한 지연.
- **15초 idle timeout이 위험하다.** 데몬은 15초 안에 명령이 안 오면 컨트롤러를
  자동 해제하고 **카메라를 정지**시킨다
  ([camera_server_daemon.py:169](../paradex/io/camera_system/camera_server_daemon.py#L169)).
  liveness는 컨트롤러의 `run()` heartbeat 스레드에 의존하는데
  ([remote_camera_controller.py:147](../paradex/io/camera_system/remote_camera_controller.py#L147)),
  이 스레드는 `executor.execute()`와 GIL을 공유한다. 길거나 블로킹되는 grasp가
  heartbeat를 굶기면 → trial 중간에 카메라가 멈춘다.

### P3. "fixed variables are bad things"
`system/current/` config로 빠져야 할 하드코딩 값들:

| 위치 | 하드코딩 | 되어야 할 것 |
|---|---|---|
| `run_auto.py:54` | `DEFAULT_PC_LIST` — **`capture4`가 조용히 빠져 있음** | pc/network config에서 유도 |
| `camera_server_daemon` | 포트 5480–5482, **15초** `RCVTIMEO`, 0.5초 heartbeat 임계 | config |
| `remote_camera_controller` | 포트, `30000/2000 ms` timeout, `sleep(0.1)` | config |
| `CameraLoader.start` | `fps=30`, `"images"`/`"videos"` 하위 폴더명 | config / 인자 |
| `camera.py` | `frame_shape=(1536,2048,3)`, 코덱 `'MJPG'` | config |
| `run_auto.py` | `sync_generator.start(fps=30)` vs `--stream_fps` (fps 두 개, desync 쉬움) | 단일 소스 |

`capture4` 누락 리터럴이 이 문제의 전형이다: 고정 리스트가 실제 리그와 어긋나는데
카메라가 "사라질" 때까지 아무도 모른다.

### P4. 프레임 유실 시 카메라 hang → 복구 불가 (가장 치명적)
LAN 케이블/패킷 유실 등으로 프레임이 끊기면 카메라가 매달리고, `run_auto`를 `pkill`
해도 다음 실행에서 카메라가 안 켜지는 일이 잦다. 원인 사슬:

1. `GetNextImage()`가 **timeout 인자 없이** 호출됨 →  프레임이 안 오면 **무한 블록**
   ([pyspin.py:180](../paradex/io/camera_system/pyspin.py#L180)). docstring도
   *"Video/Stream mode: Blocks until image available"* 라고 인정.
2. `continuous_acquire` 루프가 `get_image()`에서 멈춤 → `event["start"]`/`event["exit"]`를
   다시 못 봄 → `event["stop"]`을 영원히 set 못 함
   ([camera.py:268](../paradex/io/camera_system/camera.py#L268)).
3. `Camera.stop()`이 `event["stop"].wait()`에서 무한 대기(timeout 없음,
   [camera.py:214](../paradex/io/camera_system/camera.py#L214)),
   `Camera.end()`의 `capture_thread.join()`도 무한 대기
   ([camera.py:222](../paradex/io/camera_system/camera.py#L222)).
4. 데몬의 `CameraLoader.stop()/end()`가 이 블로킹 stop/join을 기다림 → `command_thread`
   멈춤 → 다음 컨트롤러의 register/start 불가.

**왜 pkill 후에도 안 켜지나:** `pkill run_auto`는 main PC 컨트롤러만 죽인다. capture PC의
`server_daemon.py`는 hung 스레드를 안은 채 살아있어 재실행해도 그 CameraLoader가 여전히
막혀 있다. 스레드가 Spinnaker 네이티브 `GetNextImage`에 갇히면 SIGTERM으로 안 죽어
`pkill -9`가 필요하고, 카메라가 `BeginAcquisition` 상태로 남아 "in use"가 되어 재오픈
실패 → 심하면 전원 재부팅.

> 우선순위 주의: 이 P4는 재설계(P1)와 독립적인 **버그**이며, 재-arm 최적화보다 먼저
> 고쳐야 한다. 특히 아래 4.4의 (1)은 몇 줄짜리 독립 수정이다.

---

## 3. 현재 아키텍처 (구현된 상태)

```
remote_camera_controller (main PC)
        │ ZMQ REQ/REP  register/start/stop/heartbeat/reload/end
        ▼
camera_server_daemon (capture PC)  ──>  CameraLoader  ──>  Camera ×N
                                                             │ Thread: run() → continuous_acquire()/single_acquire()
                                                             ▼
                                          sink은 루프 진입 시 `mode`로 결정됨:
                                            save_video = mode in {video, full}  → cv2.VideoWriter
                                            stream     = mode in {stream, full} → double-buffered SHM
```

오늘의 핵심 불변식: **sink 선택이 `continuous_acquire` 한 번의 수명 동안 고정된다.**
바꾸려면 ⇒ 루프 정지 ⇒ pyspin 재-arm.

---

## 4. 제안 설계

### 4.1 acquisition과 sink 분리 (핵심)

카메라는 한 번 **live**가 되면 계속 acquisition을 돌리고 **항상 SHM에 publish**한다.
recording과 스틸은 루프가 도는 중에 토글되는 sink다 — 재-arm 없음.

새 카메라 레벨 API (`start(mode, …)` 대체):

```python
cam.go_live(sync, fps, exposure_time, gain)   # acquisition 한 번 arm; SHM 항상 켜짐
cam.start_recording(path)                      # 도는 루프 안에서 VideoWriter 열기
cam.stop_recording()                           # writer 닫기; acquisition + stream은 유지
cam.grab_still(path)                           # live 버퍼에서 한 프레임 저장
cam.stop()                                     # live 상태 종료
cam.end()                                      # release
```

`continuous_acquire` 구현 스케치:
- `stream`은 live 동안 무조건 켜짐.
- `save_video`는 `recording` Event로 가드되는 런타임 플래그가 됨. `VideoWriter`를
  열고/닫는 것은 진입 시점이 아니라 토글 시 루프 안에서 일어남.
- `grab_still`은 재-arm 하는 별도 `single_acquire` 경로 대신 최신 SHM 버퍼를 읽음.

`CameraLoader`와 `remote_camera_controller`에 대응 메서드
(`go_live / start_recording / stop_recording / grab_still`)를 추가하고, 데몬에도 대응
action을 추가한다. `register/heartbeat/reload/end`는 그대로.

### 4.2 Hard-RT 조율 (P2)

- **recording을 critical path에서 뺀다.** 4.1이 있으면 trial 전에 카메라가 이미 live이고,
  `start_recording`은 그냥 "파일 열기"라 빠르고 bounded — 모션 전에 acquisition
  handshake 없음.
- **프레임 단위 정확한 trigger.** `start_recording` 시점과 모션 시작 시점의
  sync-generator frame id / trigger timestamp를 trial의 `timestamps/`에 기록한다.
  정렬이 wall-clock 근사가 아니라 프레임 정확이 된다.
- **idle timeout을 config화/일시중지 가능하게.** 데몬 15초 timeout을 config로 빼고,
  execution 동안 컨트롤러가 늘리거나 멈출 수 있게 해서 느린 grasp가 카메라 정지를
  못 일으키게 한다. 필요하면 heartbeat를 GIL-bound 스레드 밖으로 옮긴다.

### 4.3 config 기반 상수 (P3)

`system/current/`에 `camera_system` 블록 추가: 포트, idle-timeout, command timeout,
기본 fps, `frame_shape`, video 코덱, 저장 하위폴더명. `pc_list`와 PC별 serial은 기존
pc/network config에서 온다 — **앱 스크립트에 리터럴 금지**. fps 소스 하나가 stream과
sync generator 둘 다에 들어간다.

### 4.4 프레임 유실 hang 복구 (P4)

1. **(THE fix) 유한 timeout grab.** `get_image`의 `GetNextImage()`를
   `GetNextImage(grab_timeout_ms)`로 바꾼다. timeout이면 `SpinnakerException`(TIMEOUT)이
   나므로 catch → `None, frame_data` 리턴 → 루프가 `event["start"]/["exit"]`를 다시 확인.
   프레임이 끊겨도 스레드가 stop/exit에 반응하게 되는 핵심 수정.
   (이미 flush용으로 [pyspin.py:246](../paradex/io/camera_system/pyspin.py#L246)에서
   `GetNextImage(1)`을 쓰고 있어 파라미터 존재는 확인됨. 몇 줄짜리 독립 수정.)
2. **stop/end 무한 대기 제거.** `Camera.stop()`의 `event["stop"].wait(timeout=…)`,
   `Camera.end()`의 `capture_thread.join(timeout=…)`. timeout 시 강제 `EndAcquisition`
   + 에러 표시로 에스컬레이트하되 데몬은 안 막히게.
3. **데몬 hard-reset + hung 감지.** N초 무프레임이면 카메라를 죽은 것으로 보고 자동으로
   `EndAcquisition → DeInit → 재초기화`. 스레드가 살아있어도 CameraLoader를 통째로 교체할
   수 있는 강제 reset action 추가.
4. **복구 스크립트/UX.** capture PC들에서 데몬 `pkill -9` + 카메라 `DeInit` 재초기화를
   한 번에 하는 `reset_cameras` 스크립트. 데몬 기동 시 카메라 `DeInit`으로 이전 "in use"
   상태 정리.

---

## 5. 호출부 마이그레이션 맵 (전부 마이그레이션)

현재 모든 호출부는 세 가지 의도 중 하나다. 아래 grep 인벤토리가 전체 작업 목록이다.

### 의도 A — "그냥 스틸 찍기" (`image`)
매핑: `grab_still(path)` (필요 시 `go_live` 후).
- `src/calibration/handeye/capture.py`
- `src/object6d/{capture,capture_template,image_remote}.py`, `src/object6d/validate/capture_test.py`
- `src/inference/{bodex,grasp_w_gui,pringles_test,grasp_eval/real}/*` (전부 `start("image", False, …)`)
- `src/capture/camera/{image,image_remote}.py`
- `src/dataset_acquisition/graphics/{image_capture,image_traj}.py`
- `src/validate/robot/inspire_left_overlay.py`, `src/validate/camera_system/{camera,camera_loader}.py`
- AutoDex: `src/validation/perception/scene.py`, `src/execution_prev/run_demo.py`

### 의도 B — "소비자용 live stream" (`stream`)
매핑: `go_live(...)` (SHM 항상 켜짐).
- `src/calibration/extrinsic/capture.py`
- `src/capture/camera/{stream,stream_remote}.py`
- `src/validate/camera_system/camera_reader.py`
- AutoDex: `init_interactive.py`, `track_interactive.py`, 그리고
  `reorient_drop.py`, `naive_drop.py`, `reset_test.py`, `run_auto.py`의 항상-켜짐 베이스라인

### 의도 C — "액션 중 video 녹화" (`video` / `full`)
매핑: `go_live(...)` 한 번 후 `start_recording(path)` / `stop_recording()`.
- `src/capture/camera/{video,video_remote}.py`
- `src/validate/camera_system/{camera_sync,remote_camera_controller,camera_loader}.py` (`full`)
- `paradex/dataset_acqusition/capture.py` (`CaptureSession`, `start(mode, True, …)`)
- AutoDex: `run_auto.py`, `reset_test.py`, `naive_drop.py`, `reorient_drop.py` (`full`)

`CaptureSession`([paradex/dataset_acqusition/capture.py:82](../paradex/dataset_acqusition/capture.py#L82))이
가장 레버리지 큰 마이그레이션이다: 모든 `src/dataset_acquisition/*` 파이프라인의
기반이라, 이거 하나 포팅하면 여러 leaf 앱이 한 번에 커버된다.

---

## 6. 단계별 계획 (각 단계 독립 배포 가능)

0. **Hang 복구 (P4) — 최우선, 재설계와 독립.**
   - ✅ (1) 유한 timeout grab — `get_image()`가 `GetNextImage(GRAB_TIMEOUT_MS)`로 바뀌어
     timeout 시 `(None, None)` 리턴 → 루프가 start/exit 이벤트 재확인. `single_acquire`는
     bounded retry로 보강.
   - ✅ (2) stop/end 무한 대기 제거 — `Camera.stop(timeout)`/`end(timeout)`이 유한 대기 후
     경고만 남기고 데몬을 안 막음.
   - ✅ (4) capture PC용 리셋 도구 — `src/camera/reset_cameras.py` (main PC에서 실행,
     각 capture PC에 `pkill -9` 후 `server_daemon.py` 재기동).
   - ⬜ (3) 데몬 hung 감지(N초 무프레임 자동 복구) — 데몬에 스레드 추가라 실기검증 전엔
     보류. reset_cameras.py가 수동 복구는 커버.
   - ⚠️ 하드웨어 검증 필요: `src/validate/camera_system/`로 sync=True + LAN 뽑기 재현 테스트.
   - (겸사겸사) `load_timestamp_monitor`의 `exposure_time`→`exposure` 키 버그 수정.
1. **Camera core (P1).** `Camera`에 `recording` Event + 런타임 VideoWriter 토글과
   `grab_still` 추가; live 동안 SHM 항상 켜짐. 단일 PC에서
   `src/validate/camera_system/camera.py`로 검증.
2. **Loader + daemon + controller API (P1).** `go_live / start_recording /
   stop_recording / grab_still` 노출; 데몬 action 추가.
   `src/validate/camera_system/{camera_loader,remote_camera_controller}.py`로 검증.
3. **Config (P3).** 포트/timeout/fps/frame_shape/코덱/하위폴더를 `system/current/`로 이동;
   `pc_list`/serial을 config에서 유도. `capture4` 누락 수정.
4. **Hard-RT (P2).** recording/motion 시작 시 sync-trigger 로깅 추가; idle timeout을
   config화 + 일시중지 가능하게.
5. **호출부 마이그레이션** — 의도 버킷 순서(A→B→C), `CaptureSession` 먼저, 다음
   `src/capture/camera`, 그다음 AutoDex 루프들; `mode` API 삭제.
6. **문서.** `src/camera`, `src/capture` README/CLAUDE + 랜딩 페이지 label 섹션을
   새 API로 업데이트.

## 7. 리스크 & 검증

- **테스트 스위트 없음** — 각 단계 후 `src/validate/camera_system/*`로 검증 (단일 PC
  `camera.py`/`camera_loader.py`, 그다음 분산 `remote_camera_controller.py`).
- **sync 회귀** — 1/2단계는 반드시 `syncMode=True` + UTGE900로 확인하고,
  `continuous_acquire`의 frame-drop 카운터를 관찰.
- **빅뱅 마이그레이션 리스크** — 1~4단계는 내부적으로 하위호환. `mode` API 제거(5단계)만
  breaking 단계이며, 모든 버킷이 포팅된 뒤 한 번에 들어가야 함.
- **AutoDex는 다운스트림** — 호출부(5절)들이 paradex를 직접 import 하므로 5단계와
  동시에(lockstep) 마이그레이션해야 함.
