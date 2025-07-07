

# SPDX-License-Identifier: MIT
import time
import cv2
import onnxruntime as ort
import argparse
import numpy as np
import sys
from dependencies.box_utils import predict
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face detection using UltraFace-320 onnx model
face_detector_onnx = "models/version-RFB-320.onnx"
face_detector = ort.InferenceSession(face_detector_onnx)

# Scale current rectangle to square box
def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width) / 2)
    dy = int((maximum - height) / 2)
    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes

# Face blurring using pixelation
def blur_area_pixelate(image, box, blocks=10):
    (x1, y1, x2, y2) = [max(0, int(b)) for b in box]  # sanitize box coordinates
    face = image[y1:y2, x1:x2]
    h, w = face.shape[:2]
    if h == 0 or w == 0:
        return image  # skip invalid box
    temp = cv2.resize(face, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = pixelated
    return image

# Face detection method
def faceDetector(orig_image, threshold=0.7):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0).astype(np.float32)

    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs

# ------------------------------------------------------------------------------------------------------------------------------------------------


def process_video_with_model(video_path, output_path="output_blurred_video.mp4", blocks=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    # Get original video info
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Input Video: {video_path}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Resolution: {width}x{height}")
    print(f"[INFO] Total Frames: {total_frames}")
    print("[INFO] Starting video processing...")

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (width, height))

    # Start timing
    start_time = time.time()

    processed_frames = 0
    frame_times = []     # Store per-frame processing times
    frame_indices = []   # Store corresponding frame indices

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()

        boxes, labels, probs = faceDetector(frame)
        for i in range(boxes.shape[0]):
            box = scale(boxes[i, :])
            frame = blur_area_pixelate(frame, box, blocks)

        out.write(frame)
        cv2.imshow("Video - Blurred Faces", frame)

        frame_end = time.time()
        frame_time = frame_end - frame_start

        frame_times.append(frame_time)
        frame_indices.append(processed_frames)

        processed_frames += 1

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_frame = total_time / processed_frames if processed_frames else 0

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[DONE] Output saved to: {output_path}")
    print(f"[TIME STATS] Total processing time: {total_time:.2f} seconds")
    print(f"[TIME STATS] Average time per frame: {avg_time_per_frame:.4f} seconds")

    # Plotting the latency graph
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, frame_times, label="Latency per Frame", color="blue")
    plt.xlabel("Frame Index")
    plt.ylabel("Processing Time (s)")
    plt.title("Frame Index vs Processing Time (Latency)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("frame_latency_plot.png")
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=False, help="input image")
    parser.add_argument("-v", "--video", type=str, required=False, help="input video")
    args = parser.parse_args()

    if args.video:
        process_video_with_model(args.video)
        return

    img_path = args.image if args.image else "dependencies/1.jpg"

    orig_image = cv2.imread(img_path)
    if orig_image is None:
        print(f"[ERROR] Could not load image: {img_path}")
        sys.exit(1)

    boxes, labels, probs = faceDetector(orig_image)

    for i in range(boxes.shape[0]):
        box = scale(boxes[i, :])
        orig_image = blur_area_pixelate(orig_image, box, blocks=10)

    cv2.imshow('Privacy Enforced - Blurred Faces', orig_image)

    print("[INFO] Press ESC to close the window.")
    while True:
        key = cv2.waitKey(10)
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting...")
        cv2.destroyAllWindows()
        sys.exit(0)






