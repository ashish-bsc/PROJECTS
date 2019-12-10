from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64 # Batch Size for training
epochs = 10 # Number of epochs to train for
latent_dim = 256 # Latent dimensionality of the encoding space
num_samples = 10000 # Number of the samples to trai on

path_1 = 'human_text.txt'
path_2 = 'robot_text.txt'
human_texts = []
robot_texts = []
human_characters = set()
robot_characters = set()
with open(path_1, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min (num_samples, len(lines) - 1)]:
    human_text = line
    human_texts.append(human_text)
    for char in human_text:
        if char not in human_characters:
            human_characters.add(char)
with open(path_2, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min (num_samples, len(lines) - 1)]:
    robot_text = line
    robot_texts.append(robot_text)
    for char in robot_text:
        if char not in robot_characters:
            robot_characters.add(char)
            
human_characters = sorted(list(human_characters))
robot_characters = sorted(list(robot_characters))
num_encoder_tokens = len(human_characters)
num_decoder_tokens = len(robot_characters)
max_encoder_seq_length = max([len(txt) for txt in human_texts])
max_decoder_seq_length = max([len(txt) for txt in robot_texts])

print("Number of samples :", len(human_texts))
print("Number of unique input tokens :", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence legth of inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

human_token_index = dict(
        [(char, i) for i, char in enumerate(human_characters)])
robot_token_index = dict(
       [(char,i) for i, char in enumerate(robot_characters)])

encoder_human_data = np.zeros(
        (len(human_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_human_data = np.zeros(
        (len(human_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_robot_data = np.zeros(
        (len(human_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (human_text, robot_text) in enumerate(zip(human_texts, robot_texts)):
    for t, char in enumerate(human_text):
        encoder_human_data[i,t, human_token_index[char]] = 1.
    for t, char in enumerate(robot_text):
        decoder_human_data[i, t, robot_token_index[char]] = 1.
        if t>0:
            decoder_robot_data[i, t-1, robot_token_index[char]]=1.
            
encoder_humans = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state = True)
encoder_robots, state_h, state_c = encoder(encoder_humans)

encoder_states = [state_h, state_c]

decoder_humans = Input(shape=(None, num_decoder_tokens))

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_robots, _, _ = decoder_lstm(decoder_humans,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_robots = decoder_dense(decoder_robots)

model = Model([encoder_humans, decoder_humans], decoder_robots)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_human_data, decoder_human_data], decoder_robot_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('bot.h5')





