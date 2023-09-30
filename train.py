import json
import os
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Check if the 'models' directory exists, and create it if not
if not os.path.exists('models'):
    os.makedirs('models')

# Load and preprocess data
with open('data/processed/train.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

titles = []
contents = []

for line in lines:
    title, content = line.strip().split("$@$")
    titles.append(title)
    contents.append(content)

# Tokenize text
tokenizer_title = tf.keras.preprocessing.text.Tokenizer()
tokenizer_title.fit_on_texts(titles)

# Add '<SOS>' and '<EOS>' tokens to the tokenizer
tokenizer_title.word_index['<SOS>'] = len(tokenizer_title.word_index) + 1
tokenizer_title.word_index['<EOS>'] = len(tokenizer_title.word_index) + 2

sequences_title = tokenizer_title.texts_to_sequences(titles)

tokenizer_content = tf.keras.preprocessing.text.Tokenizer()
tokenizer_content.fit_on_texts(contents)
sequences_content = tokenizer_content.texts_to_sequences(contents)

# Save the tokenizers to files
with open('models/tokenizer_title.pickle', 'wb') as handle:
    pickle.dump(tokenizer_title, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('models/tokenizer_content.pickle', 'wb') as handle:
    pickle.dump(tokenizer_content, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Pad sequences
max_title_length = max(len(seq) for seq in sequences_title)
max_content_length = max(len(seq) for seq in sequences_content)

padded_sequences_title = tf.keras.preprocessing.sequence.pad_sequences(sequences_title, maxlen=max_title_length, padding='post')
padded_sequences_content = tf.keras.preprocessing.sequence.pad_sequences(sequences_content, maxlen=max_content_length, padding='post')

# Save the pad sequences in a config.json file
config = {
    'max_title_length': max_title_length,
    'max_content_length': max_content_length,
}

with open('models/config.json', 'w') as file:
    json.dump(config, file)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences_content, padded_sequences_title, test_size=0.1, random_state=42)

# Define these variables based on your data and model configuration
input_vocab_size = len(tokenizer_content.word_index) + 1   # Vocabulary size for content
output_vocab_size = len(tokenizer_title.word_index) + 1    # Vocabulary size for titles
embedding_dim = 128  # Adjust as needed
lstm_units = 256     # Adjust as needed

# Define the encoder
encoder_inputs = tf.keras.layers.Input(shape=(max_content_length,))
encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(units=lstm_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Define the decoder
decoder_inputs = tf.keras.layers.Input(shape=(max_title_length,))
decoder_embedding = tf.keras.layers.Embedding(output_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(units=lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Create the Seq2Seq model
model = tf.keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

# Compile the model with custom loss function for sequence to sequence tasks
def custom_loss(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

# Define hyperparameters
num_epochs = 20  # Increase the number of epochs
batch_size = 64

# Train the model
model.fit([X_train, y_train], y_train, epochs=num_epochs, batch_size=batch_size, validation_data=([X_val, y_val], y_val))

# Save the trained model in the native Keras format
model.save('models/title_generation_model')
