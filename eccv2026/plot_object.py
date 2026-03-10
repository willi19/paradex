import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. 데이터 로드
df = pd.read_csv('./eccv2026/pose_distance_summary.csv')

# 2. X축 설정 (4, 8, 15, 21개 뷰)
x_values = [4, 8, 15, 21]
x_labels = ["4", "8", "15", "21"]
# 데이터 컬럼명 매핑
cols = ['mean_vertex_distance_fixed4', 'mean_vertex_distance_fixed8', 
        'mean_vertex_distance_fixed15', 'mean_vertex_distance_all']

# 3. 논문용 스타일 및 폰트 설정 (STIX 사용)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 300
})

plt.figure(figsize=(10, 6))

# 4. 객체별 데이터 플로팅
# 객체가 21개로 많으므로 다양한 색상을 사용합니다.
colors = sns.color_palette("husl", n_colors=len(df))

for i, row in df.iterrows():
    values = row[cols].values.astype(float) * 1000.0
    plt.plot(x_values, values, 
             label="_nolegend_",
             marker='o', 
             markersize=4, 
             alpha=0.7, 
             linewidth=1.2,
             color=colors[i])

# 4-1. 전체 객체 평균(Overall) 라인 추가 (각 view별 평균)
overall_values = df[cols].astype(float).mean(axis=0).values * 1000.0

print(overall_values)
plt.plot(
    x_values,
    overall_values,
    label="Overall",
    marker='.',
    markersize=6,
    linewidth=2.5,
    linestyle="--",
    color="black",
    alpha=1.0,
    zorder=5,
)

# 5. 축 및 레이아웃 설정
plt.xlabel("Number of Camera Views", fontweight='normal')
plt.ylabel("Mean Vertex Distance (Pose Consistency) (mm)", fontweight='normal')
plt.xticks(x_values, x_labels)

# Y축의 값 차이가 클 경우 로그 스케일을 고려할 수 있습니다. 
# 일반 스케일로 보려면 아래 줄을 주석 처리하세요.
# plt.yscale('log') 

plt.grid(True, linestyle='--', alpha=0.5)

# 범례는 Overall만 표시하고, 그래프 내부에 배치합니다.
plt.legend(loc="upper right", frameon=True, edgecolor="black")
plt.tight_layout(rect=[0, 0, 0.8, 1])

# 6. 파일 저장
plt.savefig("pose_consistency_by_object.pdf", bbox_inches='tight')
plt.savefig("pose_consistency_by_object.png", dpi=300, bbox_inches='tight')

plt.show()
