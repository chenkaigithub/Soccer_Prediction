import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop

# Prepare data for the neural net

def prepare_batch(games_mat):
    start_year = 5
    year_column = games_mat[:,0].reshape(-1,1) - start_year
    home_mat = np_utils.to_categorical(games_mat[:,1])
    away_mat = np_utils.to_categorical(games_mat[:,2])
    x = np.hstack([year_column, home_mat, away_mat])

    dif_column = games_mat[:,3]
    dif_column[dif_column > 0] = 1 # Home wins
    dif_column[dif_column < 0] = 2 # Away wins
    y = np_utils.to_categorical(dif_column)

    return (x, y)

origMat = np.genfromtxt("games.csv", delimiter=',', dtype='int')
np.random.shuffle(origMat)

pieces = np.vsplit(origMat, 6)

(train_x, train_y) = prepare_batch(np.vstack(pieces[:4]))
(test_x, test_y) = prepare_batch(pieces[4])
(valid_x, valid_y) = prepare_batch(pieces[5])

# Use neural net on data

batch_size = 64
nb_epoch = 50

model = Sequential([
    Dense(64, input_dim=train_x.shape[1]),
    Activation('relu'),
    Dropout(0.5),
    Dense(64),
    Activation('relu'),
    Dropout(0.5),
    Dense(3),
    Activation('softmax'),
])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['accuracy'])

history = model.fit(train_x, train_y,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(valid_x, valid_y))

(score, accuracy) = model.evaluate(test_x, test_y, verbose=0)
print('Test score:', score)
print('Test accuracy:', accuracy)