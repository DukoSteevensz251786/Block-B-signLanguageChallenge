Sign Language Recognition System
A real-time sign language recognition application that uses computer vision and machine learning to detect and classify hand signs from A-Z.
📋 Features

Real-time hand detection and tracking using MediaPipe
Sign language alphabet recognition (A-Z)
Live webcam feed with visual feedback
Confidence-based prediction display
Lightweight and portable executable

🚀 Quick Start
For Users (Running the Application)

Download SignLanguageRecognition.exe
Double-click to run (no installation needed!)
Allow camera access when prompted
Show hand signs to the camera
Press ESC to exit

For Developers (Running from Source)
Prerequisites

Python 3.8 or higher
Webcam

Installation

Clone or download this repository
Install dependencies:

bashpip install -r requirements.txt

Run the application:

bashpython Practice.py
📦 Dependencies

opencv-python - Computer vision and camera handling
mediapipe - Hand detection and landmark tracking
numpy - Numerical operations
scikit-learn - Machine learning model (Random Forest)
pyinstaller - For creating standalone executable

🛠️ Building the Executable
To create a standalone .exe file:
Windows:
bashpyinstaller --onefile --windowed --add-data "models;models" --collect-all sklearn --collect-all mediapipe --name "SignLanguageRecognition" scripts/Practice.py
Linux/Mac:
bashpyinstaller --onefile --windowed --add-data "models:models" --collect-all sklearn --collect-all mediapipe --name "SignLanguageRecognition" Practice.py
The executable will be created in the dist folder.

💡 How It Works

Hand Detection: MediaPipe detects hand landmarks (21 points) in real-time
Feature Extraction: Extracts 3D coordinates (x, y, z) for each landmark (63 features total)
Classification: Random Forest classifier predicts the sign letter
Confidence Scoring:

High confidence (>70%): Shows the predicted letter
Medium confidence (30-70%): Shows prediction with "?" and confidence score
Low confidence (<30%): Shows "Not recognised"



🎯 Usage Tips

Ensure good lighting conditions
Position your hand clearly in frame
Keep hand stable for better recognition
The model works best with one hand at a time
Minimum detection confidence: 60%

⚙️ Configuration
You can adjust these parameters in Practice.py:
pythonhands = mp_hands.Hands(
    max_num_hands=1,               # Number of hands to detect
    min_detection_confidence=0.6,  # Detection threshold
    min_tracking_confidence=0.6    # Tracking threshold
)

# Confidence thresholds
if prob > 0.7:      # High confidence
if prob > 0.3:      # Medium confidence
🔧 Troubleshooting
Application won't start

Make sure your webcam is connected and not in use by another application
Check if camera permissions are granted
Try running as administrator

Poor recognition accuracy

Improve lighting conditions
Position hand closer to camera
Ensure hand is fully visible in frame
Keep hand steady

Antivirus blocking the .exe

This is a false positive common with PyInstaller executables
Add an exception in your antivirus software
Or run from source code instead

Camera not found error

Check if webcam is connected
Try changing camera index in code: cv2.VideoCapture(1) instead of cv2.VideoCapture(0)
Ensure no other application is using the camera

📊 Model Information

Algorithm: Random Forest Classifier
Input Features: 63 (21 landmarks × 3 coordinates)
Output Classes: 26 (A-Z alphabet)
Model File: models/sign_model_rf.pkl

🔒 System Requirements
For Executable:

Windows 7/8/10/11 (64-bit)
Webcam
~500 MB disk space
4 GB RAM (recommended)

For Development:

Python 3.8+
Same as above

📝 License
This project is created for educational purposes.
👥 Contributing
Contributions are welcome! Feel free to:

Report bugs
Suggest new features
Submit pull requests
Improve documentation

📧 Support
If you encounter any issues:

Check the troubleshooting section
Ensure all dependencies are installed correctly
Verify your webcam is working properly

🎓 Academic Use
This project can be used as a reference for:

Computer vision applications
Real-time hand gesture recognition
Machine learning classification
Human-computer interaction projects

⚠️ Known Limitations

Works best with clear, well-lit environments
May struggle with similar hand positions
One hand detection only
Requires consistent hand positioning during training and inference

🔄 Version History

v1.0 - Initial release with A-Z recognition

🙏 Acknowledgments

MediaPipe by Google for hand detection
OpenCV community for computer vision tools
scikit-learn for machine learning algorithms
