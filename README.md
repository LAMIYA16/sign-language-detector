#  Real-Time Sign Language Detection

A real-time sign language detection system built using a Convolutional Neural Network (CNN) with TensorFlow, OpenCV, and a Streamlit-based web interface. This project detects American Sign Language (ASL) alphabets from live webcam input.

---

## 🚀 Features

-  Real-time hand gesture detection via webcam  
-  Deep learning model trained on ASL Alphabet dataset  
-  Simple and interactive Streamlit web interface  
-  Modular codebase for easy development and deployment  

---

## 📁 Project Structure
sign-language-detector/ 



├── app.py # Streamlit frontend application


├── real_time_detection.py # Real-time prediction backend using webcam


├── train_model.py # Script to train the CNN model


├── model/


│ └── model.h5 # Trained model file


├── asl_alphabet_train/ # ASL training dataset


├── asl_alphabet_test/ # ASL testing dataset


├── requirements.txt # List of dependencies


└── README.md # Project documentation


## 💡 How It Works


- The camera captures hand gestures within a fixed rectangle.

- The gesture is preprocessed and passed to the CNN model.

- The predicted ASL alphabet is displayed on-screen in real-time.

## Training the Model
     python train_model.py
## Run the Application
    streamlit run app.py

## Dataset

- ASL Alphabet Dataset:

  
🔗 Download from Kaggle 

## Improvements Needed:
- Add support for real ASL words (using WLASL dataset)
- Improve model accuracy with more training








