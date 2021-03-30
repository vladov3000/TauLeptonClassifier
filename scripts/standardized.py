import utils
import sklearn
import sklearn.metrics as metrics
from sklearn.ensemble import GradientBoostingClassifier
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
 
def train_model(n_estimators=100, max_depth=4, models_path="../models", images_path="../images"):
    X_train, Y_train, X_test, Y_test = utils.load_split_data()
    
    model_path = f"{models_path}/scaled_{n_estimators}_{max_depth}_model.pkl"
    if os.path.exists(model_path):
        print("Using existing model")
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
    else:
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
        model.fit(X_train, Y_train)
        with open(model_path, "wb+") as model_file:
            pickle.dump(model, model_file)

    conif_model = utils.synth_model(model, bit_width='ap_fixed<11,6>', build=False)
    profile_fig = conif_model.profile()
    results = { 
        f"Gradient BDT Est {n_estimators} D{max_depth}": utils.get_stats(model, X_test, Y_test),
        f"Conifer Gradient BDT": utils.get_stats(conif_model, X_test, Y_test)
    }
    utils.make_plot(results)

def main():
    train_model()

if __name__=="__main__":
    main()
