from collections import defaultdict
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import re
import tensorflow as tf
import keras
from keras import layers
from dataset import dftrain, dfval, dftest
from eval import *
from sklearn.preprocessing import LabelEncoder
from PIL import ImageFont
import visualkeras
import keras_nlp
import jieba
from fugashi import Tagger
from hanziconv import HanziConv
from pythainlp.tokenize import word_tokenize
import unicodedata

def identify_script(text):
    scripts = {
        'Latin': range(0x0000, 0x024F),
        'Cyrillic': range(0x0400, 0x04FF),
        'CJK': range(0x4E00, 0x9FFF),
        'Thai': range(0x0E00, 0x0E7F),
        'Arabic': range(0x0600, 0x06FF),
        'Devanagari': range(0x0900, 0x097F)
    }
    
    script_counts = {script: 0 for script in scripts}
    for char in text:
        code = ord(char)
        for script, range_set in scripts.items():
            if code in range_set:
                script_counts[script] += 1
                break
    
    return max(script_counts, key=script_counts.get)

def preprocess_text(text, lang='unknown'):
    if lang == 'unknown':
        script = identify_script(text)
    else:
        script = lang

    if script == 'CJK':
        # Detect if it's Chinese or Japanese
        if any(unicodedata.name(char).startswith('CJK UNIFIED IDEOGRAPH') for char in text):
            # Convert traditional to simplified Chinese
            text = HanziConv.toSimplified(text)
            # Tokenize Chinese text
            return ' '.join(jieba.cut(text))
        else:
            # Assume it's Japanese
            japanese_tagger = Tagger('-Owakati')
            return ' '.join([word.surface for word in japanese_tagger(text)])
    elif script == 'Thai':
        # Thai tokenization with PyThaiNLP
        return ' '.join(word_tokenize(text))
    else:
        # For other scripts, use a general cleaning method
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

# Apply preprocessing
dftrain['text'] = dftrain.apply(lambda row: preprocess_text(row['text'], row['labels']), axis=1)
dfval['text'] = dfval.apply(lambda row: preprocess_text(row['text'], row['labels']), axis=1)
dftest['text'] = dftest.apply(lambda row: preprocess_text(row['text'], row['labels']), axis=1)

print(dftest)
# Create and configure the TextVectorization layer
max_length = 50  # As per your model input shape
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=10000,  # Adjust based on your vocabulary size
    output_sequence_length=max_length,
    standardize=None  # We're using our own cleaning function
)

# Adapt the layer to the training data only
vectorize_layer.adapt(dftrain['text'].values)

# Convert text to sequences for all datasets
trainseq = vectorize_layer(dftrain['text'].values).numpy()
valseq = vectorize_layer(dfval['text'].values).numpy()
testseq = vectorize_layer(dftest['text'].values).numpy()

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(dftrain['labels'])  # Fit on training data only
encoded_labels_train = label_encoder.transform(dftrain['labels'])
encoded_labels_val = label_encoder.transform(dfval['labels'])
encoded_labels_test = label_encoder.transform(dftest['labels'])




########################################Model##########################################
model = keras.models.Sequential([
    keras.Input(shape=(max_length,)),
    layers.Embedding(input_dim=len(vectorize_layer.get_vocabulary()), output_dim=256),
    layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(256)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])
print(model.summary())
#visualkeras
color_map = defaultdict(dict)
color_map['Dense']['fill'] = 'orange'
color_map['Embedding']['fill'] = 'blue'
color_map['Bidirectional LSTM']['fill'] = 'pink'



# Load a font for labeling (optional)
try:
    font = ImageFont.truetype("arial.ttf", 12)  # Adjust path as needed
except IOError:
    font = None  # Use default if custom font not found

# Create a layered view of the model with custom colors and labels
visualkeras.layered_view(
    model,
    to_file='output.png',
    color_map=color_map,
    legend=True,
    font=font
    
).show()


# Loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # Ensure from_logits is False
    metrics=['accuracy']
)

# Training
batch_size = 64
epochs = 5

model.fit(trainseq, encoded_labels_train, validation_data=(valseq, encoded_labels_val), 
          batch_size=batch_size, epochs=epochs, verbose=2)

model.save('language_detection_model.h5')

# Save the vectorize_layer and label_encoder
tf.saved_model.save(vectorize_layer, 'vectorize_layer')
import joblib
joblib.dump(label_encoder, 'label_encoder.joblib')

########################eval###################################

test_loss, test_accuracy = model.evaluate(testseq, encoded_labels_test)
print(f"Test accuracy: {test_accuracy}")

# Make predictions
train_predictions = model.predict(trainseq)
val_predictions = model.predict(valseq)
test_predictions = model.predict(testseq)

# Convert probabilities to class labels
train_predicted_labels = label_encoder.inverse_transform(train_predictions.argmax(axis=1))
val_predicted_labels = label_encoder.inverse_transform(val_predictions.argmax(axis=1))
test_predicted_labels = label_encoder.inverse_transform(test_predictions.argmax(axis=1))

# Populate the 'prediction' columns
dftrain['prediction'] = train_predicted_labels
dfval['prediction'] = val_predicted_labels
dftest['prediction'] = test_predicted_labels


# Create the confusion matrix
cm = confusion_matrix(dftest['labels'], dftest['prediction'])
confusion_df = pd.DataFrame(cm, 
                            index=label_encoder.classes_, 
                            columns=label_encoder.classes_)
# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', cbar=True)

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.tight_layout()
plt.show()


def detect_language(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text, 'unknown')
    
    # Vectorize the text
    vectorized_text = vectorize_layer([preprocessed_text]).numpy()
    
    # Make prediction
    prediction = model.predict(vectorized_text)
    
    # Get the predicted label
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
interactive_language_detection()