import time
import threading

# This shared dictionary will be updated by face and object detectors
assistant_state = {
    "people_seen": set(),
    "objects_detected": set(),
    "last_command": ""
}

# Lock to safely update state across threads
state_lock = threading.Lock()

def update_faces(names):
    with state_lock:
        for name in names:
            assistant_state["people_seen"].add(name)

def update_objects(labels):
    with state_lock:
        for label in labels:
            assistant_state["objects_detected"].add(label)

def reset_dynamic_state():
    with state_lock:
        assistant_state["people_seen"].clear()
        assistant_state["objects_detected"].clear()

def get_state_summary():
    with state_lock:
        people = ", ".join(assistant_state["people_seen"]) or "no one I recognize"
        objects = ", ".join(assistant_state["objects_detected"]) or "nothing specific"
        return f"I see {people}. I also detect {objects}."

def handle_command(text, speak_fn, gpt_instance=None):
    #print(assistant_state)

    text = text.lower()
    with state_lock:
        assistant_state["last_command"] = text

    if "who" in text and "here" in text:
        people = ", ".join(assistant_state["people_seen"]) or "no one I recognize"
        speak_fn(f"I see {people}.")

    elif "what" in text and ("see" in text or "around" in text or "object" in text):
        objects = ", ".join(assistant_state["objects_detected"]) or "nothing specific"
        speak_fn(f"I detect {objects} around you.")

    elif gpt_instance:
        response = gpt_instance.get_gpt_chat_response(text)
        speak_fn(response)

    else:
        speak_fn("I'm not sure how to respond to that.")
