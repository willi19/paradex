#!/bin/bash

# 경로 설정
BASE_DIR="/home/temp_id/shared_data/capture/hri_openarm/bottle/8/video_extracted"
OUTPUT_DIR="/home/temp_id/shared_data/capture/hri_openarm/bottle/8/grid"
mkdir -p "$OUTPUT_DIR"

# 수정된 부분: .mp4 같은 파일을 제외하고 오직 '디렉토리'만 가져옵니다.
cd "$BASE_DIR"
CAM_DIRS=($(ls -vd */ | sed 's/\///g')) 
cd - > /dev/null

# FFmpeg 입력 인자 생성
INPUTS=""
for dir in "${CAM_DIRS[@]}"; do
    # 디렉토리인 경우에만 입력값에 추가
    if [ -d "$BASE_DIR/$dir" ]; then
        INPUTS="$INPUTS -framerate 30 -start_number 101 -i $BASE_DIR/$dir/%05d.jpg"
    fi
done

# xstack 레이아웃 설정 (4x4 격자)
LAYOUT="0_0|w0_0|w0*2_0|w0*3_0|"
LAYOUT+="0_h0|w0_h0|w0*2_h0|w0*3_h0|"
LAYOUT+="0_h0*2|w0_h0*2|w0*2_h0*2|w0*3_h0*2|"
LAYOUT+="0_h0*3|w0_h0*3"

# 실행: 1080p 해상도로 인코딩
ffmpeg -y $INPUTS \
    -filter_complex "xstack=inputs=14:layout=$LAYOUT" \
    -vf "scale=1920:-2,format=yuv420p" \
    -c:v libx264 -crf 23 -preset faster \
    "$OUTPUT_DIR/grid_video_1080p.mp4"