# TFLite Object Detection Codelab (Smart Assist)

An Android application built with TensorFlow Lite and CameraX to detect Indian currency notes in real time and provide spoken feedback.  
The app is designed as an accessibility-focused prototype, with an additional weather screen and voice output support.

## Overview

This project extends the basic object-detection codelab pattern into a practical use case:
- **Live currency detection** using a bundled `.tflite` model.
- **On-screen bounding boxes** with confidence scores.
- **Text-to-Speech (TTS)** announcements of total detected amount.
- **Simple home flow** with a currency module and a weather module.

## Features

### Currency Detection
- Detects Indian note denominations: **₹10, ₹20, ₹50, ₹100, ₹200, ₹500**
- Real-time camera preview via **CameraX**
- Bounding box overlay with class label and confidence
- Non-Maximum Suppression (NMS) to reduce duplicate overlapping detections
- Start/stop camera controls
- Flashlight toggle while camera is active
- Status text showing note count and total rupee amount

### Voice Assistance
- Uses Android **TextToSpeech** API
- Announces: `Total amount is X rupees`
- Speech is throttled with delay/cooldown to reduce repeated announcements

### Weather Screen (Prototype)
- Basic weather UI with mock values
- Reads weather information aloud using TTS

## Tech Stack

- **Language:** Kotlin
- **Framework:** Android (AppCompat)
- **ML Runtime:** TensorFlow Lite (`org.tensorflow:tensorflow-lite:2.14.0`)
- **Camera:** CameraX (`1.3.0`)
- **Async:** Kotlin Coroutines
- **Minimum SDK:** 23
- **Target/Compile SDK:** 34

## Project Structure

```text
TFLite-Object-Detection-Codelab/
├── app/
│   ├── src/main/
│   │   ├── assets/
│   │   │   ├── currency_model.tflite
│   │   │   └── labels.txt
│   │   ├── java/org/tensorflow/codelabs/objectdetection/
│   │   │   ├── HomeActivity.kt
│   │   │   ├── MainActivity.kt
│   │   │   └── WeatherActivity.kt
│   │   ├── res/layout/
│   │   │   ├── activity_home.xml
│   │   │   ├── activity_camera.xml
│   │   │   └── activity_weather.xml
│   │   └── AndroidManifest.xml
│   └── build.gradle
├── build.gradle
└── settings.gradle
```

## Model Information

- Model file: `app/src/main/assets/currency_model.tflite`
- Model family: **YOLOv8n** (as reported in project notes)
- Training data: approximately **16k images** (as reported in project notes)
- Reported precision: **99.2%**

> Note: Real-world performance can vary with lighting, blur, camera angle, folded notes, and occlusions.  
> Occasional misclassification is still expected.

## Permissions

The app requests:
- `CAMERA` (for live detection)
- `INTERNET` (declared in manifest)

It also declares a TTS service query for speech capability.

## Getting Started

### Prerequisites
- Android Studio (latest stable recommended)
- Android SDK 34
- JDK 17 (recommended for modern Android Gradle Plugin workflows)
- Android device or emulator with camera support (real device strongly recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ro-lex404/TFLite-Object-Detection-Codelab.git
   ```
2. Open the project in Android Studio.
3. Let Gradle sync complete.
4. Build the app:
   ```bash
   ./gradlew assembleDebug
   ```
5. Install and run on a device:
   ```bash
   ./gradlew installDebug
   ```

## How to Use

1. Launch the app.
2. On the home screen, choose **Currency Detection**.
3. Grant camera permission when prompted.
4. Tap **Start Camera**.
5. Point camera at Indian currency notes.
6. Read detected labels and totals from the status panel.
7. Listen for spoken total amount announcements.
8. Optionally toggle **Flash** in low-light conditions.

## Current Limitations

- Detection may be unstable in poor lighting or cluttered backgrounds.
- Similar-looking notes can sometimes be misclassified.
- Weather module currently uses static/mock data.
- No multilingual speech output configuration yet.

## Roadmap Ideas

- Improved voice-first UX for low-vision users
- Gesture or hardware-button shortcuts
- Navigation assistance integration
- Obstacle detection integration
- Live weather API integration (replace mock weather data)

## Contributing

Contributions are welcome. You can help by:
- Improving model accuracy and robustness
- Optimizing inference performance
- Enhancing accessibility and spoken interaction
- Expanding automated tests and instrumentation coverage

If you plan a substantial change, open an issue first to discuss scope.
