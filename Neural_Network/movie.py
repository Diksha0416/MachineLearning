from keras.datasets import imdb
import numpy as np
(train_data, train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000)
# keeping top 10000 most frequently occuring words in the training data. Rare words will be discarded.
# in labels 0 stand for negative review and 1 stands for positive review


## for decoding the reviews back to english
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[2]])


## encoding the integer seq into a binary matrix
def vectorize_sequences(sequence, dimension=10000):
    results = np.zeros((len(sequence), dimension))
    # sets specific indices of results[i] to 1s
    for i, sequence in enumerate(sequence):
        results[i,sequence] = 1
    return results
# vectorize training and test data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

## Model definition
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

## Compiling the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

## Configuring the optimizer
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(learning_rate= 0.001), loss='binary_crossentropy', metrics=['accuracy'])

## Using custom losses and metrics
from keras import losses
from keras import metrics

## Using custom losses and metrics
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

## Setting aside a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

## Training your model
model.compile(optimizer = 'rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(partial_x_train, partial_y_train, epochs = 20, batch_size=512, validation_data=(x_val, y_val))

## Plotting the training and validation loss
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_key = 'accuracy' if 'accuracy' in history_dict else list(history_dict.keys())[2]  # Usually 'accuracy' or 'acc'
acc = history_dict[acc_key]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

## Plotting the training and validation accuracy
plt.clf()
acc_value = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs,acc_value,'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

