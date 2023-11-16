import os
import json
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class makeInput(object):
    def __init__(self, dataname):
        self._dataname = dataname
        self.input_path = f'../input/{dataname}'

    def read_data(self, filename, target_name, nrows):
        filepath = os.path.join(self.input_path, filename)
        if 'gz' in filepath:
            data = pd.read_csv(filepath, compression='gzip', nrows=nrows)
        else:
            data = pd.read_csv(filepath, nrows=nrows)
        X = data.drop(columns=[target_name])
        y = data[target_name]
        return X, y

    def save_featuremap(self, X, y):
        folder_path = "../input/FeatureMap"
        os.makedirs(folder_path, exist_ok=True)

        feature_map = {}
        feature_map['dataset'] = self._dataname
        feature_map['nrows'] = X.shape[0]
        feature_map['ncols'] = X.shape[1]
        feature_map['label'] = y.name
        feature_map['label_category'] = 'binary' if len(set(y)) == 2 else 'category'

        feature_dict = {}
        for col in X.columns:
            spec_dict = {}
            spec_dict['source'] = 'a' if 'C' in col else 'b'
            spec_dict['type'] = 'categorical' if X.dtypes[col] == object or col == 'id' or len(set(X[col])) < 15 or col[
                0] == "C" else 'numeric'
            spec_dict["padding_idx"] = 0
            spec_dict["vocab_size"] = len(set(X[col])) if spec_dict['type'] == "categorical" else 1
            spec_dict["oov_idx"] = spec_dict["vocab_size"] - 1 if spec_dict['type'] == "categorical" else None
            feature_dict[col] = spec_dict
        feature_map['features'] = feature_dict

        file_path = os.path.join(folder_path, f'{self._dataname}.json')
        with open(file_path, 'w') as json_file:
            json.dump(feature_map, json_file)
        print(f'FeatureMap Path: {file_path}')

    def make_input(self, X, y):
        file_path = f"../input/FeatureMap/{self._dataname}.json"
        if os.path.exists(file_path):
            with open(file_path, 'rb') as json_file:
                feature_map = json.load(json_file)
        else:
            print(f"Error: Feature map file '{self._dataname}.json' not found.")
            return

        feature_dictionary = feature_map['features']
        for feat in feature_dictionary:
            if feature_dictionary[feat]['type'] == 'categorical':
                lbe = LabelEncoder()
                X[feat] = lbe.fit_transform(X[feat])
            elif feature_dictionary[feat]['type'] == 'numeric':
                mms = MinMaxScaler(feature_range=(0, 1))
                X[feat] = mms.fit_transform(X[[feat]])

        X_tensor = {}
        for col in X.columns:
            X_tensor[col] = torch.tensor(X[col].values, dtype=torch.long)

        data_tensor = {}
        data_tensor['X'] = X_tensor
        data_tensor['y'] = torch.tensor(y.values, dtype=torch.long)

        data_file_path = os.path.join(self.input_path, f'{self._dataname}.pt')
        torch.save(data_tensor, data_file_path)
        print(f'Data File Path: {data_file_path}')


if __name__ == '__main__':
    mi = makeInput("avazu")
    X, y = mi.read_data('train.gz', 'click', 50000)
    mi.save_featuremap(X, y)
    mi.make_input(X, y)
