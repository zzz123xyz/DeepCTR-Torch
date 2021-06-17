import os
import sys
sys.path.append('..')
print(os.getcwd())

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DeepFM


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


if __name__ == "__main__":
    # ratio = 0.5
    ratio = 1
    # data = pd.read_csv("../../dataset/ml-1m/ratings_movies_users.csv").iloc[:, 1:] #add the .iloc[:, 1:] to drop first column
    data = pd.read_csv("../../dataset/ml-1m/ratings_movies_poster_users_08_random_train_binary_balance.csv").iloc[:, 1:] #add the .iloc[:, 1:] to drop first column
    split_location = int(np.floor(ratio*len(data)))
    data = data.iloc[:split_location]
    val_data = pd.read_csv("../../dataset/ml-1m/ratings_movies_poster_users_01_random_val_binary_balance.csv").iloc[:, 1:] #add the .iloc[:, 1:] to drop first column
    test_data = pd.read_csv("../../dataset/ml-1m/ratings_movies_poster_users_01_random_test_binary_balance.csv").iloc[:, 1:] #add the .iloc[:, 1:] to drop first column
    all_data = pd.concat([data, val_data, test_data])
    sparse_features = ["MovieId", "UserId",
                       "Gender", "Age", "Occupation", "Zip-code", ]
    target = ['Binary_rating']

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        all_data[feat] = lbe.fit_transform(all_data[feat])
        data[feat] = lbe.transform(data[feat])
        val_data[feat] = lbe.transform(val_data[feat])
        test_data[feat] = lbe.transform(test_data[feat])
    # preprocess the sequence feature

    key2index = {}
    genres_list = list(map(split, all_data['Genres'].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)
    # Notice : padding=`post`
    genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )
    train_genres_list = genres_list[:len(data)]
    val_genres_list = genres_list[len(data):len(data)+len(val_data)]
    test_genres_list = genres_list[len(data)+len(val_data):]


    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    fixlen_feature_columns = [SparseFeat(feat, all_data[feat].nunique(), embedding_dim=4)
                              for feat in sparse_features]

    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('Genres', vocabulary_size=len(
        key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean')]  # Notice : value 0 is for padding for sequence input feature

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    model_input = {name: data[name] for name in sparse_features}  #
    model_input["Genres"] = train_genres_list

    model_val_input = {name: val_data[name] for name in sparse_features}  #
    model_val_input["Genres"] = val_genres_list

    model_test_input = {name: test_data[name] for name in sparse_features}  #
    model_test_input["Genres"] = test_genres_list
    # 4.Define Model,compile and train

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
    model.compile("adam", "binary_crossentropy", metrics=['auc', 'acc', "logloss"], )
    # model.compile("adam", "binary_crossentropy", metrics=['acc', "logloss"], )
    # history = model.fit(model_input, data[target].values, batch_size=256, epochs=100, verbose=2, validation_split=0.2)
    history = model.fit(model_input, data[target].values, batch_size=256, epochs=100, verbose=2,
                        validation_data=(model_val_input, val_data[target].values))

    # for other test:
    # # model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
    # # model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )
    # history = model.fit(model_input, data[target].values, batch_size=256, epochs=100, verbose=2, validation_split=0.2)
    # # print(history)

    pred_ans = model.predict(model_test_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test_data[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test_data[target].values, pred_ans), 4))