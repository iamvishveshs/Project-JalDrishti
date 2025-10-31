# Project JalDrishti (Himachal AI Flood Alert)

A real-time, AI-powered flood detection system using a hybrid model (**YOLOv8 + Color/Turbidity Analysis**) to monitor rivers in Himachal Pradesh. It sends urgent, bilingual alerts to district-specific Telegram channels.

![Python Version](https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Library](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Library](https://img.shields.io/badge/YOLOv8-004F71?style=for-the-badge&logo=ultralytics)
![Platform](https://img.shields.io/badge/Telegram-26A5E4?style=for-the-badge&logo=telegram)

---

## Project Overview

Flash floods are a major threat in Himachal Pradesh. **JalDrishti** provides a scalable and automated solution to monitor high-risk river locations 24/7.

It uses a lightweight but effective AI model to detect floods with high accuracy, minimizing false positives and allowing it to run even on free CPU hardware (like **Streamlit Cloud**).

When a threat is confirmed, the system automatically sends a **critical bilingual alert** (English & Hindi) to private Telegram channels for each district — ensuring that residents and travelers get real-time warnings.

---

## Key Features

* **Real-Time Monitoring**
  Analyze live webcam feeds (via `streamlit-webrtc`) or pre-recorded videos.

* **Lightweight AI Detection (CPU-Friendly)**

  * **YOLOv8 Segmentation** – Detects river boundaries.
  * **Color Analysis (Turbidity)** – Measures the percentage of brown/muddy water (a flood indicator).

* **Robust Alerting System**
  Uses a moving average (`deque`) to send alerts only if a flood condition persists (e.g., 75% of frames in the last minute).

* **District-Based Subscriptions**
  Users subscribe only to districts they care about via a Telegram bot.

* **Urgent Bilingual Alerts**
  Alerts are automatically sent in both **English and Hindi**.

* **Secure Configuration**
  All tokens and channel IDs are safely managed with **Streamlit Secrets Management (`secrets.toml`)**.

---

## System Architecture

This project has **two main components:**

### 1. `app.py` (The Monitor / Detection Script)

* Built with **Streamlit**
* Reads frames from webcam or video input
* Runs YOLOv8 for river segmentation
* Performs **turbidity analysis** on river region
* Uses a **moving average** to track persistent flood frames
* Triggers alerts when flood threshold (e.g., 75%) is met
* Sends alerts via Telegram using channel IDs stored in `st.secrets`

### 2. `bot.py` (The Subscription Manager)

* Built using **python-telegram-bot**
* Manages user subscriptions
* Provides district selection buttons
* Sends private invite links for each district’s alert channel

---

## 🛠️ Technology Stack

| Category             | Technology           | Purpose                                       |
| -------------------- | -------------------- | --------------------------------------------- |
| **AI / ML**          | PyTorch              | Core deep learning framework                  |
|                      | YOLOv8-Seg           | Object segmentation model for river detection |
| **Monitoring App**   | Streamlit            | Interactive monitoring dashboard              |
|                      | OpenCV               | Frame capture & image processing              |
|                      | streamlit-webrtc     | Live webcam streaming                         |
| **Subscription Bot** | python-telegram-bot  | Telegram bot management                       |
| **Deployment**       | Streamlit Cloud      | Free hosting for `app.py`                     |
|                      | PythonAnywhere / VPS | 24/7 hosting for `bot.py`                     |
| **Utilities**        | Requests             | Send Telegram alerts via HTTP API             |

---

## 📂 Repository Structure

```
📁 Project-JalDrishti/
│
├── app.py                 # Streamlit flood detection app
├── bot.py                 # Telegram subscription bot
├── requirements.txt       # Python dependencies
├── packages.txt           # System-level dependencies (OpenCV)
├── style.css              # Fixes webcam aspect ratio
├── .gitignore             # Security rules (ignore secrets, model files)
├── models/                # Contains YOLOv8 model files (e.g., best.pt)
└── .streamlit/
    └── secrets.toml       # Secure secrets file (not uploaded)
```

---

## 🚀 Setup & Deployment

### 1. Prerequisites

* Python **3.11+**
* Telegram account
* GitHub repository
* Streamlit Cloud account (for `app.py`)
* A 24/7 hosting service for `bot.py` (PythonAnywhere, Render, or VPS)

---

### 2. Create Telegram Bot & Channels

1. **Create a Bot**
   Use [@BotFather](https://t.me/BotFather) to create your bot and save the bot token.

2. **Create Private Channels**
   For each district (e.g., `Mandi District Flood Alerts`).

3. **Add Bot as Admin**
   Give it “Post messages” permission.

4. **Get Invite Links & Channel IDs**

   * Use `@RawDataBot` to find channel IDs (e.g., `-1003213730436`).
   * Create private invite links (e.g., `t.me/+...`).

---

### 3. Local Repository Setup

```bash
# Clone your repository
git clone https://github.com/your-username/jaldrishti.git
cd jaldrishti

# Create virtual environment
python -m venv venv
source venv/bin/activate  # (Windows: .\venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

---

### 4. Configuration

#### **Configure `bot.py`**

```python
BOT_TOKEN = "YOUR_BOT_TOKEN"

DISTRICT_INVITE_LINKS = {
    "mandi": "t.me/+abcd1234",
    "shimla": "t.me/+xyz7890",
    ...
}
```

#### **Configure `app.py`**

* Set the `LOCATION_ID` for the monitoring site.
* Ensure `LOCATION_TO_DISTRICT_MAP` matches district names.
* Load your YOLOv8 model (`best.pt`) in the `load_models()` function.

#### **Create Secrets File**

`/.streamlit/secrets.toml`

```toml
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"

[district_channel_map]
mandi_district = "-100......."
kullu_district = "-100......."
shimla_district = "-100......."
bilaspur_district = "-100......."
chamba_district = "-100......."
hamirpur_district = "-100......."
kangra_district = "-100......."
lahaul_and_spiti_district = "-100......."
sirmaur_district = "-100......."
solan_district = "-100......."
una_district = "-100......."
kinnaur_district = "-100......."
```

---

### 5. Running the System

#### **A. Run the Telegram Subscription Bot**

Keep this running 24/7 on your VPS or PythonAnywhere:

```bash
python bot.py
```

*(Tip: Use `nohup python bot.py &` or a systemd service for background execution.)*

#### **B. Run the Flood Monitoring App**

Run one instance of this app for each location:

```bash
streamlit run app.py
```

---

## Future Enhancements

* Integration with Google Maps to visualize live flood locations
* Support for rainfall & temperature sensor data
* Multi-location parallel monitoring dashboard
* SMS/IVR alerts for non-smartphone users

---

## 🧑‍💻 Authors

**Vishvesh Shivam**
Student | Web Developer | Founder of [TheVsHub.in](https://www.thevshub.in)
📧 [iamvishveshs@gmail.com](mailto:iamvishveshs@gmail.com)
**Akshay Kumar**
Student | CSE[AIML]
📧 [LinkedIn](https://www.linkedin.com/in/akshaykumar0405))
**Abhay Rana**
Student | CSE[AIML]
📧 [LinkedIn](https://www.linkedin.com/in/abhayranaa/)
**Ayush Sharma**
Student | CSE[AIML]
📧 [LinkedIn](https://www.linkedin.com/in/ayush-sharma-student)
**Aayush Chauhan**
Student | CSE[AIML]
📧 [LinkedIn](https://www.linkedin.com/in/aayush-chauhan-804269303)

---

## 🪪 License

This project is released under the **MIT License**.
Feel free to use, modify, and distribute with attribution.

---

