import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop

number_of_teams = 107

# Prepare data for the neural net

def prepare_batch(games_mat):
    start_year = 5
    year_column = games_mat[:,0].reshape(-1,1) - start_year
    home_mat = np_utils.to_categorical(games_mat[:,1], nb_classes=number_of_teams)
    away_mat = np_utils.to_categorical(games_mat[:,2], nb_classes=number_of_teams)
    x = np.hstack([year_column, home_mat, away_mat])

    dif_column = games_mat[:,3]
    dif_column[dif_column > 0] = 1 # First team wins
    dif_column[dif_column < 0] = 2 # First team loses
    y = np_utils.to_categorical(dif_column, nb_classes=3)

    return (x, y)

mat = np.genfromtxt("games.csv", delimiter=',', dtype='int')
# Eliminate home vs away favoring:
n_mat = np.copy(mat[:, np.argsort([0,2,1,3])])
n_mat[:,3] *= -1
last_mat = np.vstack([mat, n_mat])

np.random.shuffle(last_mat)

pieces = np.vsplit(last_mat, 6)

(train_x, train_y) = prepare_batch(np.vstack(pieces[:4]))
(test_x, test_y) = prepare_batch(pieces[4])
(valid_x, valid_y) = prepare_batch(pieces[5])

# Use neural net on data

batch_size = 64
nb_epoch = 100

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

predicted = model.predict_classes(test_x, verbose=0)

def percentage_equal_to(arr, val):
    return (arr == val).sum() * 100 / arr.size

print("Predictions:")
print("No winner:", percentage_equal_to(predicted, 0), "%")
print("First team winner:", percentage_equal_to(predicted, 1), "%")
print("First team lost:", percentage_equal_to(predicted, 2), "%")
