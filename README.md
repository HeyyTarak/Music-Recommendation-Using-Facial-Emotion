# Music-Recommendation-Using-Facial-Emotion

Emotion-Based Music Recommendation Model
This project uses facial emotion recognition to suggest music based on detected emotions in real-time. It utilizes Python and various libraries, such as Streamlit for the interface, OpenCV (cv2) and MediaPipe for facial detection, and TensorFlow for emotion classification.

Introduction
The Emotion-Based Music Recommendation Model analyzes real-time video input to detect facial emotions and suggests music tracks accordingly. This project combines facial emotion detection with a music recommendation engine, making personalized suggestions that match the user's emotional state.

Features
Real-time emotion detection through facial analysis
Music recommendation based on detected emotions
User-friendly interface powered by Streamlit
Requirements
The following libraries and dependencies are required:

OpenCV (cv2)
NumPy
MediaPipe
TensorFlow
Streamlit
AV (for Streamlit media handling)

Project Structure
The project is organized as follows:

data_collection: Modules and scripts for collecting and annotating facial emotion data
model_training: Code and models for training the emotion detection classifier
music_recommendation: Contains the logic for mapping emotions to specific music genres or playlists
interface: Streamlit app for the user interface, handling real-time video input and displaying music suggestions

Contributing
Contributions are welcome! Feel free to submit a pull request for any changes, fixes, or improvements.
