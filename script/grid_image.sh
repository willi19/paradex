for d in /home/temp_id/shared_data/0126_ikea_purchase/20260126_182008/raw/images; do
    time_str=$(basename $(dirname $(dirname "$d")))
    
    # 1. scale=320:-1 -> 가로 320px로 리사이즈 (세로는 비율 맞춤)
    # 2. tile=5x4 -> 그리드 생성
    # 3. -q:v 5 -> JPEG 품질 설정 (1~31, 숫자가 클수록 저화질/저용량. 5~10 추천)
    ffmpeg -pattern_type glob -i "$d/*.png" \
           -vf "scale=320:-1,tile=5x4" \
           -q:v 5 \
           "/home/temp_id/shared_data/0126_ikea_purchase/grid_${time_str}.jpg"
done