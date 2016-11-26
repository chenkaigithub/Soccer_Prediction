import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from keras.layers import Input, Dense, LSTM, GRU, Dropout, merge
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.utils.np_utils import to_categorical
from keras.models import Model

timesteps = 5

def to_index(team1, team2):
    return str(min(team1, team2)) + "-" + str(max(team1, team2))

games = pd.read_csv("games.csv")
games["game_index"] = games.index
grouped_both = games.groupby(by=lambda i: to_index(games.get_value(i, "home"), games.get_value(i, "away")))

def group_to_sequences(data):
    return extract_patches_2d(data.as_matrix(), (timesteps + 1, len(games.columns)))

mat = np.concatenate([group_to_sequences(data) for g, data in grouped_both if len(data) > timesteps])
mat[:,1:,0] -= mat[:,:-1,0]
mat[:,0,0] = 0

chosen_games = mat[:,-1,:]
def get_side_sequences(column):
    column_num = 1 if column == "home" else 2
    grouped = games.sort_values(by=[column, "game_index"]).groupby([column])
    max_length = grouped["game_index"].agg(["count"]).max().iloc[0]
    def get_padded_mat(data):
        res = data.as_matrix().copy()
        res.resize((max_length, len(games.columns)))
        return res
    games_by_team = np.array([get_padded_mat(data) for g, data in grouped])
    index_chosen_games = [np.where(games_by_team[r[column_num],:,4] == r[4])[0] for r in chosen_games]
    index_timestep_games = np.tile(index_chosen_games, timesteps) - np.arange(start=timesteps, stop=0, step=-1)
    return games_by_team[chosen_games[:,column_num][:, None], index_timestep_games, 3, np.newaxis]

home_sequences = get_side_sequences("home")
away_sequences = get_side_sequences("away")
both_sequences = mat[:,:-1,[0,3]].copy()
y = mat[:,-1,3].copy()
y = to_categorical((y>0)*1 + (y<0)*2, nb_classes=3)

input_both = Input(shape=(timesteps,2))
input_home = Input(shape=(timesteps,1))
input_away = Input(shape=(timesteps,1))

encoded_both = LSTM(32)(input_both)
shared_RNN = LSTM(32)
encoded_home = shared_RNN(input_home)
encoded_away = shared_RNN(input_away)

merged = merge([encoded_both, encoded_home, encoded_away], mode="concat")
dropout = Dropout(0.5)(merged)
dense = Dense(32, activation="relu")(dropout)
dropout = Dropout(0.5)(dense)
predictions = Dense(3, activation='softmax')(dropout)

model = Model(input=[input_both, input_home, input_away], output=predictions)

model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([both_sequences, home_sequences, away_sequences], y, nb_epoch=20, validation_split=0.2)

plt.hist(np.argmax(model.predict([both_sequences, home_sequences, away_sequences]), axis=-1))
plt.show()
