import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_points(points, title="3D Points", color='blue', size=20, alpha=0.6, 
                   subsample=None, show_axes=True, figsize=(10, 8)):
    """
    3D points를 간단하게 시각화하는 함수
    
    Args:
        points: Nx3 array of 3D points
        title: 제목
        color: 점 색상 ('blue', 'red', 'green' 등 또는 RGB tuple)
        size: 점 크기
        alpha: 투명도 (0-1)
        subsample: 너무 많으면 N개만 랜덤 샘플링 (None이면 모든 점 표시)
        show_axes: 축 표시 여부
        figsize: figure 크기
    """
    if len(points) == 0:
        print("No points to plot!")
        return
    
    # 서브샘플링
    if subsample is not None and len(points) > subsample:
        indices = np.random.choice(len(points), subsample, replace=False)
        plot_points = points[indices]
        print(f"Showing {subsample} out of {len(points)} points")
    else:
        plot_points = points
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 점들 그리기
    ax.scatter(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2], 
              c=color, s=size, alpha=alpha)
    
    # 축 설정
    if show_axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    ax.set_title(f"{title}\n({len(plot_points)} points)")
    
    # 축 비율 맞추기
    max_range = np.array([plot_points[:,0].max()-plot_points[:,0].min(),
                         plot_points[:,1].max()-plot_points[:,1].min(),
                         plot_points[:,2].max()-plot_points[:,2].min()]).max() / 2.0
    mid_x = (plot_points[:,0].max()+plot_points[:,0].min()) * 0.5
    mid_y = (plot_points[:,1].max()+plot_points[:,1].min()) * 0.5
    mid_z = (plot_points[:,2].max()+plot_points[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()


def plot_3d_points_colored(points, colors=None, title="3D Points", size=20, alpha=0.6, 
                          subsample=None, colormap='viridis', figsize=(10, 8)):
    """
    색상이 있는 3D points 시각화
    
    Args:
        points: Nx3 array
        colors: N개의 색상 값 (None이면 Z값으로 색칠)
        colormap: matplotlib colormap 이름
    """
    if len(points) == 0:
        print("No points to plot!")
        return
    
    # 서브샘플링
    if subsample is not None and len(points) > subsample:
        indices = np.random.choice(len(points), subsample, replace=False)
        plot_points = points[indices]
        plot_colors = colors[indices] if colors is not None else None
    else:
        plot_points = points
        plot_colors = colors
    
    # 색상 설정
    if plot_colors is None:
        plot_colors = plot_points[:, 2]  # Z값으로 색칠
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2], 
                        c=plot_colors, cmap=colormap, s=size, alpha=alpha)
    
    # 컬러바 추가
    plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}\n({len(plot_points)} points)")
    
    plt.show()


def plot_multiple_3d_points(points_list, labels=None, colors=None, title="Multiple 3D Point Sets", 
                           size=20, alpha=0.6, figsize=(12, 8)):
    """
    여러 개의 3D point set을 한번에 시각화
    
    Args:
        points_list: list of Nx3 arrays
        labels: list of labels for each point set
        colors: list of colors for each point set
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 기본 색상들
    default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, points in enumerate(points_list):
        if len(points) == 0:
            continue
            
        color = colors[i] if colors else default_colors[i % len(default_colors)]
        label = labels[i] if labels else f"Set {i+1}"
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=color, s=size, alpha=alpha, label=label)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    plt.show()
