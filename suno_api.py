import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SUNO_API_KEY")

BASE_URL = "https://api.sunoapi.org/api/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- Helper functions ---
def get(endpoint, params=None):
    resp = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params)
    return resp.status_code, resp.json()

def post(endpoint, payload):
    resp = requests.post(f"{BASE_URL}{endpoint}", headers=HEADERS, json=payload)
    return resp.status_code, resp.json()

# --- API Wrappers ---
def get_credits():
    status, data = get("/generate/credit")
    print("Credits status:", status)
    return data

def get_task_info(task_id):
    status, data = get("/generate/record-info", params={"taskId": task_id})
    print(f"Task {task_id} status:", status)
    return data

def get_timestamped_lyrics(task_id, audio_id):
    status, data = post(
        "/generate/get-timestamped-lyrics",
        {"taskId": task_id, "audioId": audio_id}
    )
    print(f"Timestamped lyrics for {task_id} / {audio_id} status:", status)
    return data

def get_cover_info():
    status, data = get("/suno/cover/record-info")
    print("Cover record info status:", status)
    return data

def get_lyrics_info():
    status, data = get("/lyrics/record-info")
    print("Lyrics record info status:", status)
    return data

# --- Main ---
if __name__ == "__main__":
    # 1) Credits
    credits = get_credits()
    print("Current credits:", credits)

    # 2) Example stored task IDs
    stored_ids = [
        "your-task-id-1",
        "your-task-id-2",
    ]

    all_song_data = {}
    for tid in stored_ids:
        info = get_task_info(tid)
        all_song_data[tid] = info

    print("Fetched song data for stored tasks:")
    print(all_song_data)

    # 3) Example: get timestamped lyrics (replace with real audioId)
    # lyrics = get_timestamped_lyrics("your-task-id-1", "your-audio-id")
    # print("Timestamped lyrics data:", lyrics)

    # 4) Example: get cover info
    # cover_info = get_cover_info()
    # print("Cover info data:", cover_info)

    # 5) Example: get lyrics generation info
    # lyrics_info = get_lyrics_info()
    # print("Lyrics info data:", lyrics_info)
