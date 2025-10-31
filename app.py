import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms
import tempfile
import os
import time
import threading
import requests
from collections import deque
import sys

st.set_page_config(page_title="Flood Detection System", layout="wide")

LOCATION_ID = "Sundernagar"

LOCATION_TO_DISTRICT_MAP = {
    # Mandi District
    "Sundernagar": "Mandi District",
    "Mandi Town": "Mandi District",
    "IIT Mandi (Kmaand)": "Mandi District",
    "Rewalsar Lake": "Mandi District",
    "Tattapani": "Mandi District",
    # Kullu District
    "Manali": "Kullu District",
    "Kullu Town": "Kullu District",
    "Kasol": "Kullu District",
    "Manikaran": "Kullu District",
    "Naggar": "Kullu District",
    "Bhuntar": "Kullu District",
    "Banjar/Tirthan Valley": "Kullu District",
    # Shimla District
    "Shimla Mall Road": "Shimla District",
    "Rampur Bushahr": "Shimla District",
    "Rohru": "Shimla District",
    "Narkanda": "Shimla District",
    # Chamba District
    "Chamba Town": "Chamba District",
    "Dalhousie/Khajjiar": "Chamba District",
    "Bharmour": "Chamba District",
    "Killar (Pangi Valley)": "Chamba District",
    # Kangra District
    "Dharamshala/McLeod Ganj": "Kangra District",
    "Palampur": "Kangra District",
    "Kangra Town": "Kangra District",
    "Baijnath": "Kangra District",
    # Kinnaur District
    "Reckong Peo": "Kinnaur District",
    "Kalpa": "Kinnaur District",
    "Sangla Valley": "Kinnaur District",
    "Chitkul": "Kinnaur District",
    "Pooh": "Kinnaur District",
    # Lahaul & Spiti District
    "Keylong": "Lahaul and Spiti District",
    "Jispa": "Lahaul and Spiti District",
    "Kaza": "Lahaul and Spiti District",
    "Tabo": "Lahaul and Spiti District",
    "Udaipur (Lahaul)": "Lahaul and Spiti District",
    # Sirmaur District
    "Nahan": "Sirmaur District",
    "Paonta Sahib": "Sirmaur District",
    "Renukaji Lake": "Sirmaur District",
    # Other Districts
    "Rishikesh": "Bilaspur District",
    "Lambloo": "Hamirpur District",
    "Bhakhra Dam": "Una District",
    "Solan": "Solan District"
}


TARGET_DISTRICT_NAME = LOCATION_TO_DISTRICT_MAP.get(LOCATION_ID)
TARGET_CHANNEL_ID = None


try:
    if TARGET_DISTRICT_NAME:

        TARGET_CHANNEL_ID = st.secrets["district_channel_map"].get(TARGET_DISTRICT_NAME)


    if "TELEGRAM_BOT_TOKEN" not in st.secrets or not st.secrets["TELEGRAM_BOT_TOKEN"]:
        st.error("FATAL ERROR: 'TELEGRAM_BOT_TOKEN' not found in Streamlit Secrets!")
        st.stop()

except Exception as e:
    st.error(f"FATAL ERROR: Could not read secrets. Make sure /.streamlit/secrets.toml exists and is valid.")
    st.error(f"Details: {e}")
    st.stop()


if not TARGET_DISTRICT_NAME:
    st.error(f"FATAL ERROR: Location ID '{LOCATION_ID}' not found in LOCATION_TO_DISTRICT_MAP!")
    st.stop()
if not TARGET_CHANNEL_ID:
    st.error(f"FATAL ERROR: District '{TARGET_DISTRICT_NAME}' not found in secrets 'district_channel_map'!")
    st.stop()
if not str(TARGET_CHANNEL_ID).startswith("-100"):
     st.error(f"FATAL ERROR: Configured Channel ID '{TARGET_CHANNEL_ID}' for District '{TARGET_DISTRICT_NAME}' is invalid.")
     st.stop()

st.sidebar.success(f"📍 Monitoring: {LOCATION_ID}")
st.sidebar.info(f"📣 Alerts go to: {TARGET_DISTRICT_NAME} channel")
st.sidebar.markdown("---")



st.title(f"Real-Time Flood Detection: {LOCATION_ID}")
st.markdown("""
*Hybrid AI-based flood detection with:*
- YOLO v8 Instance Segmentation (River Detection)
- CNN Classifier (Flood Classification)
- Optical Flow Analysis (Speed Detection)
- Color Analysis (Water Turbidity)
""")

st.sidebar.header("Analysis Configuration")
mode = st.sidebar.radio("Select Mode", ["Video File", "Live Webcam"])
cnn_threshold = st.sidebar.slider("CNN Threshold", 0.3, 1.0, 0.85, 0.05)
speed_threshold = st.sidebar.slider("Speed Threshold (km/h)", 0.5, 5.0, 2.0, 0.1)
brown_threshold = st.sidebar.slider("Brown Color Threshold (%)", 5, 50, 20, 1)
frame_skip = st.sidebar.slider("Process Every N Frames", 1, 10, 2, 1, help="Higher = faster but less frequent updates")

st.sidebar.markdown("---")
st.sidebar.header("Alert Settings")
alert_cooldown = st.sidebar.slider("Alert Cooldown (seconds)", 5, 600, 60, 5, help="Minimum time between Telegram alerts")

st.sidebar.subheader("Average Frame Analysis")
history_length_frames = st.sidebar.slider("Frame History Size", 10, 500, 100, 10, help="How many recent processed frames to average? (e.g., 100)")
flood_percentage_threshold = st.sidebar.slider(
    "Flood Percentage Threshold (%)", 10, 100, 75, 5, help="Send alert if flood is detected in this % of recent frames."
)


class FloodCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(FloodCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        x = self.gap(x); x = x.view(x.size(0), -1); x = self.fc(x)
        return x


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Using the hardcoded path from your previous code
        model_yolo = YOLO(".//best.pt")
        model_cnn = FloodCNN(input_channels=3).to(device)
        model_cnn.load_state_dict(torch.load('flood_classifier_model.pth', map_location=device))
        model_cnn.eval()
        st.sidebar.success("AI Models Loaded")
        return model_yolo, model_cnn, device
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        st.error(f"Error loading models: {e}. Please check model paths.")
        return None, None, None

model_yolo, model_cnn, device = load_models()
if model_yolo is None:
    st.stop()



class AlertManager:
    def __init__(self, cooldown_seconds=10):
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = 0
    def can_alert(self):
        current_time = time.time()
        return (current_time - self.last_alert_time) >= self.cooldown_seconds
    def trigger_alert(self):
        self.last_alert_time = time.time()

if 'alert_manager' not in st.session_state:
    st.session_state.alert_manager = AlertManager(cooldown_seconds=alert_cooldown)

def send_telegram_alert(specific_location, district_name, target_channel_id, status_message):
    """Sends urgent, highlighted bilingual messages to the specified DISTRICT channel ID."""


    token = st.secrets.get("TELEGRAM_BOT_TOKEN")
    if not token:
        st.sidebar.error("Telegram Bot Token is not set in secrets!")
        return


    try:
        percent_str = status_message.split(' ')[0]
        flood_likelihood_en = f"*High Risk* ({percent_str} detected)"
        flood_likelihood_hi = f"*उच्च जोखिम* ({percent_str} पता चला)"
    except Exception:
        flood_likelihood_en = "*High Risk*"
        flood_likelihood_hi = "*उच्च जोखिम*"

    text_en = f"""🚨🌊 *CRITICAL FLOOD WARNING* 🌊🚨
‼️ *IMMEDIATE DANGER DETECTED* ‼️
Evacuate low-lying areas near the river *NOW!* Seek higher ground immediately.

*Location:* `{specific_location}`
*District:* `{district_name}`
*Flood Likelihood:* {flood_likelihood_en}

⚠️ *ACTION REQUIRED: MOVE TO SAFETY. DO NOT APPROACH THE RIVER.* ⚠️
"""
    text_hi = f"""🚨🌊 *गंभीर बाढ़ चेतावनी* 🌊🚨
‼️ *तत्काल खतरा पाया गया है* ‼️
नदी के पास के निचले इलाकों को *तुरंत* खाली करें! शीघ्र ऊँचे स्थान पर जाएँ।

*स्थान:* `{specific_location}`
*जिला:* `{district_name}`
*बाढ़ की संभावना:* {flood_likelihood_hi}

⚠️ *आवश्यक कार्रवाई: सुरक्षित स्थान पर जाएँ। नदी के पास न जाएँ।* ⚠️
"""


    url = f"https://api.telegram.org/bot{token}/sendMessage"


    try:
        payload_en = {"chat_id": target_channel_id, "text": text_en, "parse_mode": "Markdown"}
        response_en = requests.post(url, json=payload_en, timeout=10)
        if response_en.status_code == 200: st.sidebar.success(f"EN alert sent to {district_name}.")
        else: st.sidebar.warning(f"Failed EN alert ({district_name}): {response_en.text}")

        time.sleep(0.5)
        payload_hi = {"chat_id": target_channel_id, "text": text_hi, "parse_mode": "Markdown"}
        response_hi = requests.post(url, json=payload_hi, timeout=10)
        if response_hi.status_code == 200: st.sidebar.success(f"HI alert sent to {district_name}.")
        else: st.sidebar.warning(f"Failed HI alert ({district_name}): {response_hi.text}")
    except Exception as e:
        st.sidebar.error(f"Error sending Telegram alert: {e}")

def trigger_alert(specific_location, district_name, target_channel_id, status_message):
    """Triggers bilingual alert (NO SOUND)."""
    if not st.session_state.alert_manager.can_alert():
        return

    st.session_state.alert_manager.trigger_alert()

    telegram_thread = threading.Thread(
        target=send_telegram_alert,
        args=(specific_location, district_name, target_channel_id, status_message),
        daemon=True
    )
    telegram_thread.start()


def show_flood_alert():
    st.error("🚨 FLOOD DETECTED! 🚨", icon="⚠")

cnn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def classify_flood_region(image_np, mask, threshold=0.85):
    if not np.any(mask): return False, 0.0
    river_region = cv2.bitwise_and(image_np, image_np, mask=mask.astype(np.uint8))
    image_rgb = cv2.cvtColor(river_region, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_normalized = image_resized.astype('float32') / 255.0
    image_tensor = cnn_transform(image_normalized).unsqueeze(0).to(device)
    with torch.no_grad(): confidence = model_cnn(image_tensor).cpu().numpy()[0][0]
    return confidence > threshold, float(confidence)

def get_merged_mask(results, frame_shape):
    mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        merged = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
        for m in masks:
            m_resized = cv2.resize(m, (frame_shape[1], frame_shape[0]))
            merged = np.maximum(merged, m_resized)
        mask = (merged > 0.5).astype(np.uint8)
    return mask

def process_frame(frame, mask, speed_kmh, confidence, status, color):
    display_frame = frame.copy()
    if np.any(mask):
        mask_colored = np.zeros_like(frame); mask_colored[:, :, 0] = mask * 255
        mask_colored[:, :, 1] = mask * 200; mask_colored[:, :, 2] = mask * 100
        display_frame = cv2.addWeighted(frame, 1, mask_colored, 0.3, 0)
    cv2.putText(display_frame, f"Speed: {speed_kmh:.2f} km/h", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(display_frame, status, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(display_frame, f"Likelihood: {confidence:.1%}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return display_frame


if mode == "Video File":
    st.subheader("Upload Video File")
    uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    if uploaded_video:
        st.session_state.alert_manager = AlertManager(cooldown_seconds=alert_cooldown)
        frame_history = deque(maxlen=history_length_frames)
        alert_sent_in_video = False
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_video.read()); input_video_path = tmp_file.name
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.info(f"Video: {width}x{height} @ {fps:.1f} fps | History: {history_length_frames} frames | Threshold: {flood_percentage_threshold}%")
        output_video_path = f"output_flood_detection_{int(time.time())}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID"); out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        col1, col2 = st.columns([3, 1])
        with col1: frame_placeholder = st.image([])
        with col2: st.subheader("Stats"); stats_placeholder = st.empty()
        alert_placeholder = st.empty(); progress_bar = st.progress(0); status_text = st.empty()
        flood_frames = 0; normal_frames = 0; frame_num = 0; prev_gray = None
        lower_brown = np.array([5, 50, 50]); upper_brown = np.array([25, 255, 200])
        start_time = time.time()
        mask = np.zeros((height, width), dtype=np.uint8); status = "Initializing..."; color = (200, 200, 200); speed_kmh = 0.0; confidence = 0.0

        while True:
            ret, frame = cap.read();
            if not ret: break
            frame_num += 1; is_flood_frame = False
            if frame_num % frame_skip == 0:
                results = model_yolo.predict(frame, imgsz=640, verbose=False, conf=0.3)
                mask = get_merged_mask(results, frame.shape)
                if prev_gray is None: prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_masked = flow * mask[..., None]; status = "Normal"; color = (0, 255, 0); speed_kmh = 0.0; confidence = 0.0
                if np.any(mask):
                    mag, ang = cv2.cartToPolar(flow_masked[..., 0], flow_masked[..., 1]); mean_speed_px = np.mean(mag[mask > 0])
                    speed_kmh = mean_speed_px * 0.05 * fps * 3.6
                    river_pixels = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8))
                    hsv = cv2.cvtColor(river_pixels, cv2.COLOR_BGR2HSV); brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
                    total_river = np.sum(mask); brown_ratio = np.sum(brown_mask > 0) / total_river * 100 if total_river > 0 else 0
                    cnn_is_flood, confidence = classify_flood_region(frame, mask, threshold=cnn_threshold)
                    if cnn_is_flood: status = "Flood Detected"; color = (0, 0, 255); flood_frames += 1; is_flood_frame = True
                    elif speed_kmh > speed_threshold and brown_ratio > brown_threshold: status = "Flood Detected"; color = (0, 0, 255); flood_frames += 1; is_flood_frame = True
                    else: status = "Normal"; color = (0, 255, 0); normal_frames += 1
                prev_gray = gray; frame_history.append(is_flood_frame)
                if len(frame_history) == history_length_frames:
                    flood_count = sum(frame_history); current_flood_pct = (flood_count / history_length_frames) * 100
                    status_text.write(f"Frame: {frame_num}/{frame_count} | Flood % (last {history_length_frames} frames): {current_flood_pct:.0f}%")
                    if current_flood_pct >= flood_percentage_threshold and not alert_sent_in_video:
                        alert_sent_in_video = True
                        with alert_placeholder.container(): show_flood_alert()
                        trigger_alert(
                            specific_location=LOCATION_ID,
                            district_name=TARGET_DISTRICT_NAME,
                            target_channel_id=TARGET_CHANNEL_ID,
                            status_message=f"{current_flood_pct:.0f}% of recent frames"
                        )
                    elif current_flood_pct < (flood_percentage_threshold - 20):
                        alert_sent_in_video = False; alert_placeholder.empty()
            display_frame = process_frame(frame, mask, speed_kmh, confidence, status, color); out.write(display_frame)
            if frame_num % 5 == 0: frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            if frame_num % 10 == 0:
                with stats_placeholder.container():
                    st.metric("Flood", flood_frames); st.metric("Normal", normal_frames)
                    if (flood_frames + normal_frames) > 0: rate = (flood_frames / (flood_frames + normal_frames)) * 100; st.metric("Rate", f"{rate:.1f}%")
            progress = frame_num / frame_count; progress_bar.progress(progress)
            if frame_num % frame_skip != 0 or len(frame_history) < history_length_frames:
                elapsed = time.time() - start_time; fps_actual = frame_num / elapsed if elapsed > 0 else 0
                eta = (frame_count - frame_num) / fps_actual if fps_actual > 0 else 0
                status_text.write(f"Frame: {frame_num}/{frame_count} | {progress*100:.1f}% | {fps_actual:.1f} fps | ETA: {eta:.0f}s")
        cap.release(); out.release(); elapsed_total = time.time() - start_time

        st.divider(); st.subheader("Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Frames", frame_num)
        with col2:
            st.metric("Flood Frames", flood_frames)
        with col3:
            st.metric("Normal Frames", normal_frames)
        with col4:
            if (flood_frames + normal_frames) > 0:
                rate = (flood_frames / (flood_frames + normal_frames)) * 100
                st.metric("Flood Rate", f"{rate:.1f}%")
            else:
                st.metric("Flood Rate", "0.0%")

        st.write(f"Processing time: {elapsed_total:.1f} seconds")
        if alert_sent_in_video: st.error("⚠ FLOOD WAS DETECTED IN THIS VIDEO")
        else: st.success("✓ No flood detected in this video")
        st.subheader("Download Processed Video")
        if os.path.exists(output_video_path):
            with open(output_video_path, "rb") as video_file:
                st.download_button(label="Download Processed Video", data=video_file.read(), file_name=output_video_path, mime="video/avi")
            st.success(f"Video saved: {output_video_path}")
        try: os.unlink(input_video_path)
        except Exception as e: st.warning(f"Could not delete temp file {input_video_path}: {e}")


else:
    st.subheader("Live Webcam Detection")
    col1, col2 = st.columns([3, 1])
    with col1: frame_placeholder = st.image([])
    with col2: st.subheader("Live Stats"); stats_placeholder = st.empty()
    alert_placeholder = st.empty(); status_text_webcam = st.empty()
    start_button = st.button("Start Real-Time Detection", key="start_detection")
    stop_button = st.button("Stop Detection", key="stop_detection")
    if start_button:
        st.session_state.detecting = True
        st.session_state.alert_manager = AlertManager(cooldown_seconds=alert_cooldown)
        st.session_state.frame_history = deque(maxlen=history_length_frames)
        st.session_state.alert_sent = False
    if stop_button: st.session_state.detecting = False
    if "detecting" not in st.session_state: st.session_state.detecting = False
    if st.session_state.detecting:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): st.error("Cannot access webcam")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30)
            fps = cap.get(cv2.CAP_PROP_FPS); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            st.info(f"Webcam: {width}x{height} @ {fps} fps | History: {history_length_frames} frames | Threshold: {flood_percentage_threshold}%")
            flood_frames = 0; normal_frames = 0; frame_num = 0; prev_gray = None
            lower_brown = np.array([5, 50, 50]); upper_brown = np.array([25, 255, 200])
            start_time = time.time()
            mask = np.zeros((height, width), dtype=np.uint8); status = "Initializing..."; color = (200, 200, 200); speed_kmh = 0.0; confidence = 0.0

            while st.session_state.detecting:
                ret, frame = cap.read();
                if not ret: st.error("Failed to capture frame"); break
                frame_num += 1; is_flood_frame = False
                if frame_num % frame_skip == 0:
                    results = model_yolo.predict(frame, imgsz=640, verbose=False, conf=0.3)
                    mask = get_merged_mask(results, frame.shape)
                    if prev_gray is None: prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flow_masked = flow * mask[..., None]; status = "Normal"; color = (0, 255, 0); speed_kmh = 0.0; confidence = 0.0
                    if np.any(mask):
                        mag, ang = cv2.cartToPolar(flow_masked[..., 0], flow_masked[..., 1]); mean_speed_px = np.mean(mag[mask > 0])
                        speed_kmh = mean_speed_px * 0.05 * fps * 3.6
                        river_pixels = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8))
                        hsv = cv2.cvtColor(river_pixels, cv2.COLOR_BGR2HSV); brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
                        total_river = np.sum(mask); brown_ratio = np.sum(brown_mask > 0) / total_river * 100 if total_river > 0 else 0
                        cnn_is_flood, confidence = classify_flood_region(frame, mask, threshold=cnn_threshold)
                        if cnn_is_flood: status = "Flood Detected"; color = (0, 0, 255); flood_frames += 1; is_flood_frame = True
                        elif speed_kmh > speed_threshold and brown_ratio > brown_threshold: status = "Flood Detected"; color = (0, 0, 255); flood_frames += 1; is_flood_frame = True
                        else: status = "Normal River"; color = (0, 255, 0); normal_frames += 1
                    prev_gray = gray
                    if 'frame_history' not in st.session_state: st.session_state.frame_history = deque(maxlen=history_length_frames)
                    st.session_state.frame_history.append(is_flood_frame)
                    if len(st.session_state.frame_history) == history_length_frames:
                        flood_count = sum(st.session_state.frame_history); current_flood_pct = (flood_count / history_length_frames) * 100
                        elapsed = time.time() - start_time; fps_actual = (frame_num / frame_skip) / elapsed if elapsed > 0 else 0
                        status_text_webcam.write(f"Frame: {frame_num} | {fps_actual:.1f} fps | Flood %: {current_flood_pct:.0f}%")
                        if current_flood_pct >= flood_percentage_threshold:
                            if not st.session_state.alert_sent:
                                st.session_state.alert_sent = True
                                with alert_placeholder.container(): show_flood_alert()
                                trigger_alert(
                                    specific_location=LOCATION_ID,
                                    district_name=TARGET_DISTRICT_NAME,
                                    target_channel_id=TARGET_CHANNEL_ID,
                                    status_message=f"{current_flood_pct:.0f}% of recent frames"
                                )
                        elif current_flood_pct < (flood_percentage_threshold - 20):
                            st.session_state.alert_sent = False; alert_placeholder.empty()
                display_frame = process_frame(frame, mask, speed_kmh, confidence, status, color)
                frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                with stats_placeholder.container():
                    st.metric("Flood Frames", flood_frames); st.metric("Normal Frames", normal_frames)
                    if (flood_frames + normal_frames) > 0: rate = (flood_frames / (flood_frames + normal_frames)) * 100; st.metric("Flood Rate", f"{rate:.1f}%")
                if 'frame_history' in st.session_state and len(st.session_state.frame_history) < history_length_frames:
                    elapsed = time.time() - start_time; fps_actual = (frame_num / frame_skip) / elapsed if elapsed > 0 else 0
                    status_text_webcam.write(f"Frame: {frame_num} | {fps_actual:.1f} fps | Status: {status} (Buffering history...)")
            cap.release(); st.info("Detection stopped")

st.divider()
st.markdown("""
*System Performance (Sample):*
- Test Accuracy: 96.39%
- AUC Score: 99.62%
- Detection Methods: CNN + Speed + Color Analysis
""")
