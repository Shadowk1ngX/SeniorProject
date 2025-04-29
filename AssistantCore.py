import time
import threading
import cv2
import numpy as np
import datetime


# This shared dictionary will be updated by face and object detectors
assistant_state = {
    "people_seen": set(),
    "objects_detected": set(),
    "last_command": ""
}

# Lock to safely update state across threads
state_lock = threading.Lock()

def update_faces(names, state, lock):
    with lock:
        for name in names:
            #if name != "Unknown" and name not in state["people_seen"]:
            if name not in state["people_seen"]:
                state["people_seen"].append(name)
                print("AddedName:", name)
                print("Current seen:", list(state["people_seen"]))


def update_objects(labels):
    with state_lock:
        for label in labels:
            assistant_state["objects_detected"].add(label)

def reset_dynamic_state():
    with state_lock:
        assistant_state["people_seen"].clear()
        assistant_state["objects_detected"].clear()

def get_state_summary(assistant_state, lock):
    with lock:
        people = ", ".join(assistant_state["people_seen"]) or "no one I recognize"
        return f"I see {people}."
    

def capture_snapshot_from_shared_frame(shared_frame, frame_lock, filename="snapshot.png"):
    frame_shape = (480, 640, 3)  # match your actual frame shape

    with frame_lock:
        frame_np = np.ctypeslib.as_array(shared_frame.get_obj()).copy().reshape(frame_shape)
        #print(frame_np)

    if frame_np is None or frame_np.size == 0 or np.mean(frame_np) < 5:
        print("❌ Shared frame is empty or too dark to be valid.")
        return None

    cv2.imwrite(filename, frame_np)
    print(f"📸 Snapshot saved as {filename}")
    return filename

def format_names(names):
    seen = sorted(names)
    if not seen:
        people = "no one I recognize"
    elif len(seen) == 1:
        people = seen[0]
    elif len(seen) == 2:
        people = f"{seen[0]} and {seen[1]}"
    else:
        people = ", ".join(seen[:-1]) + f", and {seen[-1]}"
    return people


def handle_command(text, gpt_instance=None, assistant_state=None, lock=None, shared_frame=None, frame_lock=None, tts_queue = None):

    with lock:
        #print("Current people seen:", assistant_state["people_seen"])
        text = text.lower().strip()
    with lock:
        assistant_state["last_command"] = text

   # if "what" in text and "time" in text:
     #   Time = time.localtime()

    # “Who’s here?”
    if "who" in text and ("here" in text or "see" in text):
        with lock:
            text = format_names(assistant_state["people_seen"])
        tts_queue.put(f"I see {text}.")
        return

    # “What do you see?”
    if "what" in text and ("see" in text or "around" in text or "object" in text):
        image_path = capture_snapshot_from_shared_frame(shared_frame, frame_lock)
        text = format_names(assistant_state["people_seen"])
        if image_path and gpt_instance:
            desc = gpt_instance.get_gpt_chat_response(
                f"What do you see in this image?",
                image_path=image_path
            )
            tts_queue.put(desc)
        else:
            tts_queue.put("I’m having trouble seeing right now.")
        return

    # Fallback to GPT-powered chat
    if gpt_instance:
        response = gpt_instance.get_gpt_chat_response(text)
        tts_queue.put(response)
    else:
        tts_queue.put("I’m not sure how to respond to that.")
