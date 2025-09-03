import numpy as np
import os
import time
from scipy.spatial.transform import Rotation
import trimesh
import copy
from pathlib import Path
from tqdm.auto import tqdm

from paradex.utils.file_io import shared_dir, download_dir, eef_calib_path, load_latest_eef, rsc_path, get_robot_urdf_path
from paradex.robot.mimic_joint import parse_inspire

# Open3D 렌더러 임포트 (위에서 만든 클래스)
from paradex.visualization.open3d_viewer import Open3DVideoRenderer  # 위의 클래스를 별도 파일로 저장했다고 가정

hand_name = "allegro"
arm_name = "xarm"
obj_name = "book"

# 메시 로드
mesh = trimesh.load(os.path.join(rsc_path, "object", obj_name, obj_name+".obj"))
LINK2WRIST = load_latest_eef()

demo_path = os.path.join(shared_dir, "capture", "lookup", obj_name)
demo_name_list = os.listdir(demo_path)

# 렌더링 설정
RENDER_CONFIG = {
    "width": 1920,
    "height": 1080, 
    "fps": 30,
    "camera_eye": [-0.3, 0.0, 0.2],  # 카메라 위치
    "output_dir": os.path.join(download_dir, "rendered_videos", obj_name)
}

def process_demo_data(demo_name, demo_type):
    """단일 데모 데이터 처리"""
    try:
        # 데이터 로드
        hand_qpos = np.load(os.path.join(demo_path, str(demo_name), f"{demo_type}_hand.npy"))
        wrist_pos = np.load(os.path.join(demo_path, str(demo_name), f"{demo_type}_action.npy"))
        obj_pos = np.load(os.path.join(demo_path, str(demo_name), f"{demo_type}_objT.npy"))
        obj_pose = obj_pos[0].copy() if demo_type == "pick" else obj_pos[-1].copy()
        
        # 오브젝트 위치 정규화
        obj_pos[:,:2, 3] -= obj_pose[:2, 3]
        obj_pose[:2, 3] = 0
        
        T = wrist_pos.shape[0]
        action = np.zeros((T, 22 if hand_name == "allegro" else "inspire"))
        
        # 액션 데이터 변환
        for i in range(T):
            wrist_pos[i] = obj_pose @ wrist_pos[i] @ LINK2WRIST
            euler = Rotation.from_matrix(wrist_pos[i,:3,:3]).as_euler('zyx')
            
            action[i, 5] = euler[0]
            action[i, 4] = euler[1] 
            action[i, 3] = euler[2]
            
        action[:,:3] = wrist_pos[:, :3, 3]
        
        if hand_name == "inspire":
            hand_qpos = parse_inspire(hand_qpos)
        action[:, 6:] = hand_qpos
        
        return {
            'action': action,
            'obj_pos': obj_pos,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'action': None,
            'obj_pos': None, 
            'success': False,
            'error': str(e)
        }

def render_single_video(demo_name, demo_type, action_data, obj_pos_data, output_path):
    """단일 비디오 렌더링"""
    try:
        print(f"🎬 Rendering {demo_name}/{demo_type}...")
        
        renderer = Open3DVideoRenderer(
            obj_mesh=copy.deepcopy(mesh),
            obj_T=obj_pos_data,
            urdf_path=get_robot_urdf_path(None, hand_name),
            qpos=action_data,
            width=RENDER_CONFIG["width"],
            height=RENDER_CONFIG["height"],
            fps=RENDER_CONFIG["fps"]
        )
        
        renderer.render_video(
            output_path=output_path,
            camera_eye=RENDER_CONFIG["camera_eye"]
        )
        
        return True, None
        
    except Exception as e:
        error_msg = f"Error rendering {demo_name}/{demo_type}: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg

def batch_render_all_demos():
    """모든 데모에 대해 배치 렌더링"""
    
    # 출력 디렉토리 생성
    output_dir = Path(RENDER_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting batch rendering for {obj_name}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Render settings: {RENDER_CONFIG['width']}x{RENDER_CONFIG['height']} @ {RENDER_CONFIG['fps']}fps")
    print(f"📋 Found demos: {demo_name_list}")
    print("-" * 60)
    
    success_count = 0
    total_count = 0
    failed_renders = []
    
    # 모든 데모에 대해 반복
    for demo_name in tqdm(demo_name_list, desc="Processing demos"):
        for demo_type in ["pick", "place"]:
            total_count += 1
            
            # 출력 파일 경로
            output_filename = f"{demo_name}_{demo_type}.mp4"
            output_path = output_dir / output_filename
            
            # # 이미 존재하는 파일 스킵 (선택적)
            # if output_path.exists():
            #     print(f"⏭️  Skipping {demo_name}/{demo_type} (already exists)")
            #     success_count += 1
            #     continue
            
            # 데이터 처리
            print(f"📤 Processing data for {demo_name}/{demo_type}...")
            processed_data = process_demo_data(demo_name, demo_type)
            
            if not processed_data['success']:
                print(f"❌ Failed to process data for {demo_name}/{demo_type}: {processed_data['error']}")
                failed_renders.append(f"{demo_name}/{demo_type} (data processing)")
                continue
            
            # 비디오 렌더링
            render_success, render_error = render_single_video(
                demo_name, 
                demo_type, 
                processed_data['action'],
                processed_data['obj_pos'],
                str(output_path)
            )
            
            if render_success:
                success_count += 1
                print(f"✅ Successfully rendered {demo_name}/{demo_type}")
            else:
                failed_renders.append(f"{demo_name}/{demo_type} (rendering)")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("🎯 BATCH RENDERING SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"❌ Failed: {len(failed_renders)}")
    
    if failed_renders:
        print("\n🚨 Failed renders:")
        for failed in failed_renders:
            print(f"   - {failed}")
    
    print(f"\n📁 All videos saved to: {output_dir}")
    print("🎉 Batch rendering completed!")

def render_specific_demos(demo_names=None, demo_types=None):
    """특정 데모들만 렌더링"""
    
    if demo_names is None:
        demo_names = demo_name_list
    if demo_types is None:
        demo_types = ["pick", "place"]
    
    # 출력 디렉토리 생성
    output_dir = Path(RENDER_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Rendering specific demos: {demo_names}")
    print(f"📋 Types: {demo_types}")
    
    for demo_name in demo_names:
        if demo_name not in demo_name_list:
            print(f"⚠️  Demo '{demo_name}' not found in {demo_name_list}")
            continue
            
        for demo_type in demo_types:
            output_filename = f"{demo_name}_{demo_type}.mp4"
            output_path = output_dir / output_filename
            
            # 데이터 처리
            processed_data = process_demo_data(demo_name, demo_type)
            
            if not processed_data['success']:
                print(f"❌ Failed to process {demo_name}/{demo_type}")
                continue
            
            # 렌더링
            render_success, _ = render_single_video(
                demo_name,
                demo_type,
                processed_data['action'],
                processed_data['obj_pos'],
                str(output_path)
            )
            
            if render_success:
                print(f"✅ {demo_name}/{demo_type} completed")

# 메인 실행
if __name__ == "__main__":
    # 전체 배치 렌더링 실행
    batch_render_all_demos()
    
    # 또는 특정 데모만 렌더링하고 싶다면:
    # render_specific_demos(demo_names=["7", "8"], demo_types=["pick"])