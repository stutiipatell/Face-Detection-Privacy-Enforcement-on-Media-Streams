import cv2
import os

def get_video_info(video_path):
    if not os.path.isfile(video_path):
        print(f"[ERROR] File does not exist: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps > 0 else 0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

    print(f"\n📽️ Video Info for: {video_path}")
    print(f"🔢 Frame Rate (FPS): {fps}")
    print(f"🎞️ Total Frames: {total_frames}")
    print(f"⏱️ Duration: {duration:.2f} seconds")
    print(f"📐 Resolution: {int(width)}x{int(height)}")
    print(f"🎬 Codec: {codec_str}")
    cap.release()


# Example usage:
video_path = "/home/stuti/seminar/vid1.mp4"
get_video_info(video_path)

