import conifer
import os
import datetime
import pickle
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def open_model(filename):
    """ Returns sklearn stored in pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def synth_model(model):
    """ Return conifer model given scikit model """
    # Create a conifer config
    cfg = conifer.backends.vivadohls.auto_config()

    # Set the output directory to something unique
    cfg['OutputDir'] = 'projects/prj_{}'.format(int(datetime.datetime.now().timestamp()))
    cfg['XilinxPart'] = 'xc7vx690tffg1761-2'
    
    # Create and compile the model
    conif_model = conifer.model(model, conifer.converters.sklearn, conifer.backends.vivadohls, cfg)
    conif_model.compile()
     
    # Synthesize the model
    conif_model.build()

    return conif_model

def load_params_used(filename):
    with open(filename, "r") as f:
        return f.read().split()

def load_data(sig_filename, bg_filename, features):
    """ Returns datafram, X, Y for data stored in csv file"""
    
    sig_df = pd.read_csv(sig_filename)[features]
    bg_df = pd.read_csv(bg_filename)[features]
    
    sig_df["signal"] = 1
    bg_df["signal"] = 0
    
    df  = pd.concat([sig_df, bg_df], ignore_index=True)
    
    X = df.to_numpy()[:, :-1]
    Y = df.to_numpy()[:, -1][np.newaxis].transpose()
    
    return df, X, Y


def split_data(X, Y, test_percent=0.1):
    """ Returns X_train, Y_train, X_test, Y_test """
    total = np.concatenate((X, Y), axis=1)
    np.random.shuffle(total)
    n_X = (int) ((1 - test_percent) * total.shape[0])
    
    X_train = total[:n_X, :-1]
    X_test = total[n_X:, :-1]
    
    Y_train = total[:n_X, -1][np.newaxis].transpose()
    Y_test = total[n_X:, -1][np.newaxis].transpose()
    
    return X_train, Y_train, X_test, Y_test

def get_scaled_data(data="../data"):
    params = load_params_used(f"{data}/params5_used.txt")

    _, X, Y = load_data(f"{data}/sig.csv", f"{data}/bg.csv", params)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)
    scaler = sklearn.preprocessing.StandardScaler(copy=False)
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

    with open(f"{data}/scaler.pkl", "wb+") as scaler_file:
        pickle.dump(scaler, scaler_file)
    for i in ("X_train", "Y_train", "X_test", "Y_test"):
        np.save(f"{data}/{i}.npy", locals()[i])

    return X_train, Y_train, X_test, Y_test

def load_split_data(data="../data"):
    if not os.path.isfile(f"{data}/X_train.npy"):
        return get_scaled_data()

    res = []
    for i in ("X_train", "Y_train", "X_test", "Y_test"):
        res.append(np.load(f"{data}/{i}.npy"))
    return res
