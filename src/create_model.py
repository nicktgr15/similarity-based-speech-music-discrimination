import itertools
import pickle
import numpy as np
from sac.util import Util
import os
import feat
from sklearn.svm import SVC

DATASETS = os.path.abspath("../datasets")

features = Util.read_feature_names_from_file(os.path.join(DATASETS, "featureplans/featureplan"))

features1 = ["zcr", "flux", "spectral_rollof", "energy_stats"]
features2 = ["mfcc_stats"]
features3 = ["spectral_flatness_per_band"]
features4 = features1 + features2 + features3

data = feat.get_features(features4, DATASETS, pca=False)

TRAIN = ["mirex", "labrosa", "gtzan"]

model = SVC()
X = np.vstack((data["x_" + i] for i in TRAIN))
Y = list(itertools.chain.from_iterable([data["y_" + i] for i in TRAIN]))

model.fit(X, Y)

with open("pickled/model.pickle", 'w') as f:
    pickle.dump(model, f)
