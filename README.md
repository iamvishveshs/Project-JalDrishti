
# Himachal Pradesh AI Flood Alert System

A real-time, AI-powered flood detection system using a hybrid model (YOLOv8 + CNN + Optical Flow) to monitor rivers in Himachal Pradesh. It sends urgent, bilingual alerts to district-specific Telegram channels.

![Python Version](https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Library](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Library](https://img.shields.io/badge/YOLOv8-004F71?style=for-the-badge&logo=ultralytics)
![Platform](https://img.shields.io/badge/Telegram-26A5E4?style=for-the-badge&logo=telegram)

---

## 🎯 Project Overview

Flash floods are a major threat in Himachal Pradesh. This project provides a scalable and automated solution to monitor high-risk river locations 24/7. It uses a combination of AI models to detect floods with high accuracy, minimizing false positives.

When a threat is confirmed, the system automatically sends a **critical danger alert** in both English and Hindi to a private Telegram channel for that specific district, ensuring residents and travelers who subscribe get the warning immediately.

## ✨ Key Features

* **Real-Time Monitoring:** Analyze live webcam feeds or pre-recorded video files via a user-friendly Streamlit interface.
* **Hybrid AI Detection:** Combines four different analysis methods for high accuracy:
    1.  **YOLOv8 Segmentation:** Identifies the exact bounds of the river.
    2.  **CNN Classification:** Analyzes the river's texture and patterns.
    3.  **Optical Flow Analysis:** Tracks the water's velocity.
    4.  **Turbidity Analysis:** Measures the percentage of "brown" (mud/debris) in the water.
* **Robust Alerting:** Uses a moving average (`deque`) to trigger alerts only if a flood state persists (e.g., 75% of frames in the last minute), preventing false alarms from single-frame glitches.
* **District-Based Subscriptions:** A separate Telegram bot (`bot.py`) allows users to subscribe only to the districts they care about.
* **Urgent Bilingual Alerts:** Sends high-priority, formatted messages in both English and Hindi.

## ⚙️ System Architecture

The project is a **two-part system**:

1.  **`app.py` (The Monitor / Detection Script):**
    * This is the core detection script, built with Streamlit. It runs on a machine at the monitoring site (e.g., a PC or Jetson Nano with a camera).
    * It reads a frame, and the **YOLOv8** model generates a "mask" of the river.
    * This mask is used for three **parallel analyses**: Optical Flow (speed), CNN (classification), and Color Analysis (turbidity).
    * The results are combined into a single `is_flood_frame = True/False` decision.
    * This decision is added to a `deque` (moving average).
    * If the percentage of flood frames in the `deque` crosses a threshold (e.g., 75%), it calls the `trigger_alert()` function.
    * The alert is sent to the correct **private district channel ID** based on the script's `LOCATION_ID` setting.

2.  **`bot.py` (The Subscription Manager):**
    * This is a lightweight Telegram bot that must run **24/7** on a server (e.g., PythonAnywhere, Render, or a VPS).
    * Users find the bot and send `/start` or `/subscribe`.
    * The bot shows buttons for all available districts.
    * When a user clicks a district, the bot replies with the **private invite link** for that district's channel.
    * The user joins the channel and is now subscribed.

This "decoupled" architecture is robust: the heavy AI processing (`app.py`) is separate from the user-facing bot (`bot.py`).

ert System\n(Send Bilingual Alert to District Channel)];
    F -- Threshold Not Met --> I((Continue Monitoring));
