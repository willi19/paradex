import threading


def listen_keyboard(event_dict, stop_event=None):
    def run(event_dict, stop_event):
        print(f"Available keys: {list(event_dict.keys())}")
        while not stop_event.is_set():
            try:
                key = input().strip().lower()
                if key == 'q':
                    stop_event.set()
                    break
                elif key in event_dict:
                    event_dict[key].set()
                    print(f"[{key}] event triggered")
                else:
                    print(f"Unknown key: {key}")
            except (EOFError, KeyboardInterrupt):
                break
    if stop_event is None:
        stop_event = threading.Event()
    
    input_thread = threading.Thread(
        target=run, 
        args=(event_dict, stop_event),
        daemon=True
    )
    input_thread.start()
    return stop_event