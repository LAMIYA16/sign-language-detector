import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def run_detection():

    model = load_model('model/model.h5')


    cap = cv2.VideoCapture(0)

    img_size = 64


    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    while True:
    
       ret, frame = cap.read()
    
       if not ret:
        break
    
    
       rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
       resized_frame = cv2.resize(rgb_frame, (img_size, img_size))
    
    
       img_array = np.array(resized_frame) / 255.0  
       img_array = np.expand_dims(img_array, axis=0) 
    
    
       x1, y1, x2, y2 = 200, 200,410, 450
       roi = frame[y1:y2, x1:x2]

       cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


       resized_frame = cv2.resize(roi, (64, 64))
       img_array = np.expand_dims(resized_frame / 255.0, axis=0)
       prediction = model.predict(img_array)

    
       prediction = model.predict(img_array)
    
   
       predicted_label = labels[np.argmax(prediction)]
    
    
       cv2.putText(frame, f"Predicted: {predicted_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    
       cv2.imshow('Real-Time Sign Language Detection', frame)
    
    
       if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


    cap.release()
    cv2.destroyAllWindows()
