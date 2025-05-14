# Paralanguage Recognition for Virtual Connects

This project provides a real-time assistive system that detects emotions in individuals with cognitive or speech impairments using non-verbal cues like facial expressions, gestures, and voice tone.

## 🧠 Features
- Real-time emotion detection using CNN (for vision) and RNN (for voice tone)
- ISL (Indian Sign Language) gesture recognition
- Automatic caregiver alert when stress is detected
- Passive detection during virtual communication

## 📁 Repository Structure

| Folder / File | Description |
|---------------|-------------|
| `models/` | Trained models: ISL and Emotion Detection (`.h5` files) |
| `notebooks/` | Jupyter Notebooks for training the models |
| `src/` | Python scripts including real-time detection |
| `logs/` | Logs from interpreter or training |
| `assets/` | System architecture diagram, screenshots |
| `requirements.txt` | Python dependencies |

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/realtimedetection.py
