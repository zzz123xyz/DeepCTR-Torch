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
    # data = pd.read_csv("../../dataset/ml-1m/ratings_movies_users.csv").iloc[:, 1:] #add the .iloc[:, 1:] to drop first column
    data = pd.read_csv("../../dataset/ml-1m/ratings_movies_poster_users_train.csv").iloc[:, 1:] #add the .iloc[:, 1:] to drop first column

    sparse_features = ["MovieId", "UserId",
                       "Gender", "Age", "Occupation", "Zip-code", ]
    target = ['Rating']

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # preprocess the sequence feature

    key2index = {}
    genres_list = list(map(split, data['Genres'].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)
    # Notice : padding=`post`
    genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                              for feat in sparse_features]

    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('Genres', vocabulary_size=len(
        key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean')]  # Notice : value 0 is for padding for sequence input feature

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    model_input = {name: data[name] for name in sparse_features}  #
    model_input["Genres"] = genres_list

    # 4.Define Model,compile and train

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device)
    model.compile("adam", "mse", metrics=['mse'], )
    history = model.fit(model_input, data[target].values, batch_size=256, epochs=100, verbose=2, validation_split=0.2)

    # for other test:
    # # model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
    # # model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )
    # history = model.fit(model_input, data[target].values, batch_size=256, epochs=100, verbose=2, validation_split=0.2)
    # # print(history)
    # pred_ans = model.predict(test_model_input, 256)
    # print("")
    # print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    # print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))