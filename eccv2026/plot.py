import matplotlib.pyplot as plt
import seaborn as sns



# 1. 폰트 설정 (Times New Roman)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],  # 시스템에 설치된 Times New Roman 사용
    "mathtext.fontset": "stix",         # 수식 폰트를 Times와 유사한 STIX로 설정
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

# 데이터 재구성
x_values = [8, 12, 15, 21]
x_labels = ["8", "12", "15", "21"]
tips_data = {
    "Thumb Tip": [18.92, 14.07, 10.62, 9.29],
    "Index Tip": [6.00, 6.35, 6.48, 6.10],
    "Middle Tip": [10.04, 7.28, 6.98, 6.76],
    "Ring Tip": [7.56, 7.33, 7.81, 7.08],
    "Pinky Tip": [6.67, 8.10, 7.44, 7.16]
}

avg_values = [
    sum(per_view) / len(per_view)
    for per_view in zip(*tips_data.values())
]

plt.figure(figsize=(10, 6))
base_colors = sns.color_palette("colorblind", n_colors=5) # 논문용으로 적합한 Colorblind 팔레트
overall_color = base_colors[0]  # 기존 Thumb 색을 Overall에 사용
thumb_color = sns.color_palette("Reds", n_colors=5)[3]
colors = [thumb_color, *base_colors[1:]]

for i, (label, values) in enumerate(tips_data.items()):
    plt.plot(x_values, values, label=label, 
             marker='.', markersize=4, linewidth=1.5,
             color=colors[i], markerfacecolor=colors[i], markeredgewidth=0)

plt.plot(
    x_values,
    avg_values,
    label="Overall",
    marker='.',
    markersize=6,
    linewidth=2.5,
    linestyle="--",
    color="black",
    markerfacecolor="black",
    markeredgewidth=0,
)

# 축 및 레이블 설정
plt.xlabel("Number of Camera Views", fontweight='normal') # 보통 논문은 볼드보다 노멀 선호
plt.ylabel("Joint Estimation Error (mm)", fontweight='normal')
plt.xticks(x_values, x_labels)
plt.grid(True, linestyle='--', alpha=0.6)

plt.legend(frameon=True, edgecolor='black')
plt.tight_layout(rect=[0, 0, 0.8, 1])
# 저장
plt.savefig("hand_error_times.pdf", bbox_inches='tight')
plt.show()
