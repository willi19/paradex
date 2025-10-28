from paradex.io.camera_system.monitor_daemon import CameraMonitorDaemon



def main():
    """Main 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Camera Monitor Daemon')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Update interval in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    # Monitor 생성
    monitor = CameraMonitorDaemon(update_interval=args.interval)
    
    # 실행
    monitor.run()


if __name__ == "__main__":
    main()