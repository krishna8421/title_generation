import json
import numpy as np
import pickle
import tensorflow as tf
from train import custom_loss

# Load the saved tokenizers
with open('models/tokenizer_title.pickle', 'rb') as handle:
    tokenizer_title = pickle.load(handle)
    print("tokenizer_title loaded")

with open('models/tokenizer_content.pickle', 'rb') as handle:
    tokenizer_content = pickle.load(handle)
    print("tokenizer_content loaded")

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('models/title_generation_model', custom_objects={'custom_loss': custom_loss})
print("Model loaded")

# Load configuration from config.json
with open('models/config.json', 'r') as file:
    config = json.load(file)
    print("Config loaded")

max_content_length = config['max_content_length']
max_title_length = config['max_title_length']

def generate_title(input_text):
    # Tokenize the input text using the content tokenizer
    input_sequence = tokenizer_content.texts_to_sequences([input_text])
    input_sequence = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=max_content_length, padding='post')

    # Initialize the decoder input with a start token
    target_sequence = np.zeros((1, max_title_length))
    target_sequence[0, 0] = tokenizer_title.word_index['<SOS>']

    generated_title = []

    for _ in range(max_title_length - 1):
        # Predict the next word
        predictions = model.predict([input_sequence, target_sequence])
        predicted_word_index = np.argmax(predictions[0, _])

        if predicted_word_index == tokenizer_title.word_index['<EOS>']:
            break

        generated_title.append(tokenizer_title.index_word[predicted_word_index])

        # Update the decoder input with the predicted word
        target_sequence[0, _ + 1] = predicted_word_index

    return ' '.join(generated_title)

while True:
    input_text = input("Enter the content: ")
    generated_title = generate_title(input_text)
    print("Generated Title:", generated_title)
