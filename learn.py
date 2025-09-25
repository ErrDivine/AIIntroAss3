import os
import pickle
import time
import datetime
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
# from threadpoolctl import threadpool_limits  # OPTIONAL (see note)
import plugins

from play import AliensEnvPygame

def extract_features(observation):

    # TODO

    grid = observation
    features = []

    def cell_to_feature(cell):
        object_mapping = {
            'floor': 0,
            'wall': 1,
            'avatar': 2,
            'alien': 3,
            'bomb': 4,
            'portalSlow': 5,
            'portalFast': 6,
            'sam': 7,
            'base': 8
        }
        feature_vector = [0] * len(object_mapping)
        for obj in cell:
            index = object_mapping.get(obj, -1)
            if index >= 0:
                feature_vector[index] = 1
        return feature_vector

    for row in grid:
        for cell in row:
            cell_feature = cell_to_feature(cell)
            features.extend(cell_feature)

    return np.array(features)


def load_game_records(base_dirs, root="logs"):
    records = []
    for d in base_dirs:
        path = os.path.join(root, d, "data.pkl")
        if not os.path.exists(path):
            print(f"[skip] missing file: {path}")
            continue
        size = os.path.getsize(path)
        if size == 0:
            print(f"[skip] empty pickle: {path}")
            continue

        # Some logging frameworks append multiple pickles into one file;
        # read until EOF.
        loaded_any = False
        with open(path, "rb") as f:
            while True:
                try:
                    obj = pickle.load(f)
                    loaded_any = True
                    if isinstance(obj, list):
                        records.extend(obj)
                    else:
                        records.append(obj)
                except EOFError:
                    break
                except pickle.UnpicklingError as e:
                    print(f"[skip] unpickling error in {path}: {e}")
                    break

        if not loaded_any:
            print(f"[warn] no objects read from: {path}")

    return records


def main():
    #if specified on level then set an integer. If use all levels then set as 'all'
    level = 0
    data_list = [
        # 'game_records_lvl0_2024-xx-xx_xx-xx-xx', # 修改路径为你的数据
        # 'game_records_lvl0_2024-yy-yy_yy-yy-yy',
    ]
    data_list = plugins.get_level_logs(level)
    # data = []
    # for data_load in data_list:
    #     with open(os.path.join('logs', data_load, 'data.pkl'), 'rb') as f:
    #         data += pickle.load(f)
    data = load_game_records(data_list, root="logs")

    if not data:
        raise RuntimeError("No training data found. Check your log folders and data.pkl files.")

    X = []
    y = []
    for observation, action in data:
        features = extract_features(observation)
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)

    #Alternate models here
    clf = RandomForestClassifier(
        n_estimators=300,  # more trees benefits from parallelism
    )
    clf.fit(X, y)
    #Alternation end

    #env = AliensEnvPygame(level=0, render=False)
    #remove dir the env instance just created.
    #os.rmdir(env.log_folder)

    with open(f'models/gameplay_model_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_lvl{level}.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("模型训练完成")

if __name__ == '__main__':
    main()
