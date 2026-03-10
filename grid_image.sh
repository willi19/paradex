# 1. 대상 루트 경로 설정
BASE_DIR="/home/robot/shared_data/RSS2026_Mingi/ikea_purchase"

# 2. 각 물체(object) 폴더를 순회 (예: dust_pan, white_soap_dish 등)
for obj_dir in "$BASE_DIR"/*/; do
    # 폴더가 아닌 파일이 섞여있을 경우를 대비해 체크
    [ -d "$obj_dir" ] || continue

    obj_name=$(basename "$obj_dir")
    
    # 3. 해당 물체 폴더 내의 첫 번째 시간 폴더(time_dir) 찾기
    # 20260202_121847 같은 폴더 중 첫 번째를 가져옵니다.
    first_time_dir=$(ls -d "$obj_dir"*/ 2>/dev/null | head -n 1)

    if [ -n "$first_time_dir" ]; then
        # 이번 구조: [시간폴더]/raw/images/
        img_path="${first_time_dir}raw/images"
        
        if [ -d "$img_path" ]; then
            echo "Processing: $obj_name ($img_path)"
            
            # 4. FFmpeg 실행 (용량 최적화: 가로 320px 리사이즈 + JPG 압축)
            ffmpeg -y -pattern_type glob -i "$img_path/*.png" \
                   -vf "scale=320:-1,tile=5x4" \
                   -q:v 7 \
                   "$BASE_DIR/grid_${obj_name}.jpg"
        else
            echo "경고: $img_path 폴더를 찾을 수 없습니다."
        fi
    fi
done