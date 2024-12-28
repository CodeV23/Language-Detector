import tensorflow as tf
import keras
import joblib
from solution import preprocess_text

# Load the trained model
model = keras.models.load_model('C:\code\python\milestone\model\language_detection_model.h5')

# Load the saved vectorize layer
vectorize_layer = tf.saved_model.load('C:\code\python\milestone\model\vectorize_layer')

# Load the label_encoder
label_encoder = joblib.load('C:\code\python\milestone\model\label_encoder.joblib')

def detect_language(text):
    preprocessed_text = preprocess_text(text, 'unknown')
    vectorized_text = vectorize_layer([preprocessed_text])
    prediction = model.predict(vectorized_text)
    predicted_label = label_encoder.inverse_transform(prediction.argmax(axis=1))[0]
    return predicted_label

def interactive_language_detection():
    print("Welcome to the Language Detection Tool!")
    print("Type 'quit' to exit.")
    
    while True:
        user_input = input("\nEnter text to detect language: ")
        
        if user_input.lower() == 'quit':
            print("Thank you for using the Language Detection Tool. Goodbye!")
            break
        
        detected_language = detect_language(user_input)
        print(f"Detected language: {detected_language}")

# Run the interactive tool
if __name__ == "__main__":
    interactive_language_detection()