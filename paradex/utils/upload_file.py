import os
import subprocess
import argparse
import re
import sys
from pathlib import Path
from tqdm import tqdm

def rsync_copy(src, dst, move=False, resume=True, dry_run=False, 
               compress=False, checksum=True, verbose=False):
    """
    rsync를 사용한 안전한 파일/디렉토리 복사
    
    Args:
        src: 소스 경로
        dst: 목적지 경로
        move: 복사 후 원본 삭제
        dry_run: 실제 복사 안 하고 테스트만
        compress: 전송 중 압축 (느린 네트워크에 유용)
        checksum: 체크섬 기반 검증
        verbose: 상세 출력
    """
    src = Path(src)
    dst = Path(dst)
    
    # 소스 확인
    if not src.exists():
        print(f"❌ Source not found: {src}")
        return False
    
    # rsync 설치 확인
    if not check_rsync_installed():
        print("❌ rsync is not installed!")
        print("Install: sudo apt install rsync  (Ubuntu/Debian)")
        print("        brew install rsync       (Mac)")
        return False
    
    # 목적지 디렉토리 생성
    if dst.suffix:  # 파일인 경우
        dst.parent.mkdir(parents=True, exist_ok=True)
    else:  # 디렉토리인 경우
        dst.mkdir(parents=True, exist_ok=True)
    
    # rsync 옵션 구성
    cmd = ['rsync']
    
    # 기본 옵션
    cmd.extend([
        '-a',  # archive mode (권한, 타임스탬프 등 보존)
        '-h',  # human-readable
        '--info=progress2',  # 전체 진행률 표시
        '--partial',  # 중단된 파일 유지
    ])
    
    if move:
        cmd.append('--remove-source-files')  # 복사 후 원본 삭제
    
    if dry_run:
        cmd.append('--dry-run')
    
    if compress:
        cmd.append('-z')  # 압축 전송
    
    if checksum:
        cmd.append('--checksum')  # 체크섬 검증
    
    if verbose:
        cmd.extend(['-v', '--stats'])
    else:
        cmd.append('--quiet')  # tqdm만 보여주기
    
    # inplace: 임시 파일 없이 직접 쓰기 (재개 시 유용)
    if resume:
        cmd.append('--inplace')
    
    # 소스/목적지 추가
    cmd.append(str(src))
    cmd.append(str(dst))
    
    print(f"🚀 Starting rsync...")
    print(f"   Source: {src}")
    print(f"   Dest:   {dst}")
    if move:
        print(f"   Mode:   MOVE (will delete source)")
    if dry_run:
        print(f"   DRY RUN - no actual changes")
    print()
    
    try:
        # rsync 실행
        if verbose:
            # Verbose 모드: 직접 출력
            result = subprocess.run(cmd, check=True)
        else:
            # Progress bar 모드
            result = _rsync_with_progress(cmd)
        
        if result.returncode == 0:
            print("\n✅ Success!")
            if move and not dry_run:
                print(f"🗑️  Source removed: {src}")
            return True
        else:
            print(f"\n❌ rsync failed with code {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ rsync error: {e}")
        return False

def _rsync_with_progress(cmd):
    """
    rsync를 실행하면서 progress bar 표시
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Progress bar 초기화
    pbar = None
    total_size = None
    
    # rsync 출력 파싱
    # Format: "     12,345,678  45%  123.45MB/s    0:00:12"
    progress_pattern = re.compile(
        r'\s*([\d,]+)\s+(\d+)%\s+([\d.]+[kKmMgG]?B/s)\s+(\d+:\d+:\d+)'
    )
    
    try:
        for line in process.stdout:
            line = line.strip()
            
            if not line:
                continue
            
            # Progress line 파싱
            match = progress_pattern.search(line)
            if match:
                transferred = int(match.group(1).replace(',', ''))
                percent = int(match.group(2))
                speed = match.group(3)
                eta = match.group(4)
                
                # Total size 추정 (첫 업데이트 시)
                if total_size is None and percent > 0:
                    total_size = int(transferred * 100 / percent)
                    pbar = tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc='rsync',
                        ascii=True
                    )
                
                # Progress bar 업데이트
                if pbar:
                    pbar.n = transferred
                    pbar.set_postfix({
                        'speed': speed,
                        'eta': eta
                    })
                    pbar.refresh()
    
    finally:
        if pbar:
            pbar.close()
    
    return process.wait()

def check_rsync_installed():
    """rsync 설치 여부 확인"""
    try:
        result = subprocess.run(
            ['rsync', '--version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def get_file_size(path):
    """파일 또는 디렉토리 크기 계산"""
    path = Path(path)
    
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total
    return 0

def format_size(size):
    """바이트를 human-readable로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def estimate_time(src):
    """예상 전송 시간 계산 (참고용)"""
    size = get_file_size(src)
    
    # 네트워크 속도 가정 (1Gbps NAS)
    speed_mbps = 100  # MB/s
    
    estimated_seconds = size / (speed_mbps * 1024 * 1024)
    
    print(f"📊 Total size: {format_size(size)}")
    print(f"⏱️  Estimated time: {estimated_seconds:.1f}s (at {speed_mbps}MB/s)")
    print()
