import os
import pickle
import time
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

def main():
    #if specified on level then set an integer. If use all levels then set as 'all'
    level = 0
    data_list = [
        # 'game_records_lvl0_2024-xx-xx_xx-xx-xx', # 修改路径为你的数据
        # 'game_records_lvl0_2024-yy-yy_yy-yy-yy',
    ]
    data_list = plugins.get_level_logs(level)
    data = []
    for data_load in data_list:
        with open(os.path.join('logs', data_load, 'data.pkl'), 'rb') as f:
            data += pickle.load(f)

    X = []
    y = []
    for observation, action in data:
        features = extract_features(observation)
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)

    #Alternate models here
    clf = RandomForestClassifier(n_estimators=100)
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
