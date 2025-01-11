Todo
x
0. Calibration
1. Realtime streaming
2. Data Capture

16cam_parades/
├── README.md               # 프로젝트 설명 및 사용법
├── requirements.txt        # 필요한 라이브러리 목록
├── configs/                # 설정 파일 폴더 (예: YAML 또는 JSON)
│   └── config.yaml
├── src/                    # 소스 코드 폴더
│   ├── __init__.py         # 패키지 초기화 파일
│   ├── calibration/        # 캘리브레이션 관련 코드
│   │   ├── __init__.py
│   │   ├── checkerboard.py # 체커보드 이미지 생성/처리
│   │   ├── colmap.py       # COLMAP과의 통합 코드
│   │   └── parameters.py   # 카메라 파라미터 계산
│   ├── capture/            # 데이터 캡처 관련 코드
│   │   ├── __init__.py
│   │   ├── single_image.py # 단일 이미지 캡처 코드
│   │   └── video.py        # 동영상 스트리밍 및 녹화 코드
│   ├── streaming/          # 실시간 스트리밍 관련 코드
│   │   ├── __init__.py
│   │   └── realtime.py     # 다중 카메라 스트리밍 핸들러
│   └── utils/              # 공통 유틸리티 코드
│       ├── __init__.py
│       ├── io.py           # 파일 읽기/쓰기
│       ├── visualization.py# 시각화 도구
│       └── logger.py       # 로깅 유틸리티
├── scripts/                # 실행 스크립트
│   ├── run_calibration.py  # 캘리브레이션 실행 스크립트
│   ├── run_streaming.py    # 스트리밍 실행 스크립트
│   └── run_capture.py      # 데이터 캡처 실행 스크립트
└── tests/                  # 테스트 코드
    ├── test_calibration.py # 캘리브레이션 테스트
    ├── test_capture.py     # 캡처 모듈 테스트
    └── test_streaming.py   # 스트리밍 모듈 테스트
