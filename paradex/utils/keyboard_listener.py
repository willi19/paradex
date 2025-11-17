import threading

stop_event = threading.Event()

def listen_keyboard(event_dict):
    def run(event_dict):
        print(f"Available keys: {list(event_dict.keys())}")
        while not stop_event.is_set():
            try:
                key = input().strip().lower()
                if key in event_dict:
                    event_dict[key].set()
                    print(f"[{key}] event triggered")
                else:
                    print(f"Unknown key: {key}")
            except (EOFError, KeyboardInterrupt):
                break
    input_thread = threading.Thread(
        target=run, 
        args=(event_dict,),
        daemon=True
    )
    input_thread.start()
    
def stop_listening():
    stop_event.set()