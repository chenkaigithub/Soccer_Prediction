import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import *

number_of_teams = 107

# Prepare data for the neural net

def prepare_batch(games_mat):
    start_year = 5
    year_column = games_mat[:,0].reshape(-1,1) - start_year
    home_mat = np_utils.to_categorical(games_mat[:,1], nb_classes=number_of_teams)
    away_mat = np_utils.to_categorical(games_mat[:,2], nb_classes=number_of_teams)
    x = np.hstack([year_column, home_mat, away_mat])

    dif_column = games_mat[:,3]

    return (x, dif_column)

mat = np.genfromtxt("games.csv", delimiter=',', dtype='int')

np.random.shuffle(mat)

pieces = np.vsplit(mat, 6)

(train_x, train_y) = prepare_batch(np.vstack(pieces[:4]))
(test_x, test_y) = prepare_batch(pieces[4])
(valid_x, valid_y) = prepare_batch(pieces[5])

# Use neural net on data

batch_size = 16
nb_epoch = 15

model = Sequential([
    Dense(32, input_dim=train_x.shape[1]),
    Activation(LeakyReLU()),
    Dropout(0.5),

    Dense(32),
    Activation(LeakyReLU()),
    Dropout(0.5),

    Dense(1)
])

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=RMSprop(lr=0.0001),
              metrics=['accuracy'])

history = model.fit(train_x, train_y,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(valid_x, valid_y))

(score, accuracy) = model.evaluate(test_x, test_y, verbose=0)

print('Test score:', score)
print('Test accuracy:', accuracy)

predicted = model.predict(test_x, verbose=0)

# for (p,y) in zip(predicted,test_y):
#     print(p,y)