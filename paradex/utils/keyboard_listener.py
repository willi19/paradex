import threading

def run(event_dict):
    while True:
        key = input().strip().lower()
        if key in event_dict:
            event_dict[key].set()

def listen_keyboard(event_dict):   
    input_thread = threading.Thread(target=run, daemon=True, args=(event_dict,))
    input_thread.start()