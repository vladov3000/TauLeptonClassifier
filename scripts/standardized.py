import utils
import sklearn
import sklearn.metrics as metrics
from sklearn.ensemble import GradientBoostingClassifier
import os
import pandas as pd
import pickle
 
def train_model(n_estimators=100, max_depth=4, models_path="../models", images_path="../images"):
    X_train, Y_train, X_test, Y_test = utils.load_split_data()
    
    model_path = f"{models_path}/scaled_{n_estimators}_{max_depth}_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
    else:
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
        with open(model_path, "wb+") as model_file:
            pickle.dump(model, model_file)

    model.fit(X_train, Y_train)
    
    fig, ax = plt.subplots()
    metrics.plot_roc_curve(model, X_test, Y_test, ax=ax)
    plt.show()    
    plt.savefig(f"{images_path}/scaled_{n_estimators}_{max_depth}_model.png")

def main():
    train_model()

if __name__=="__main__":
    main()
