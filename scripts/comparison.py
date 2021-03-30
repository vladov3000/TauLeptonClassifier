import conifer
import datetime
import pickle
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

def ada_grad_conif_compare():
    grad_model = open_model("../models/GradientBoosted_params5_trees50_depth2.pkl")
    ada_model = open_model("../models/params5_trees800_depth4.pkl")
    conif_model = synth_model(grad_model, bit_width='ap_fixed<11,6>', build=False)

    _, _, X_test, Y_test = load_split_data() 

    results = {
        "Gradient BDT": get_stats(grad_model, X_test, Y_test),
        "AdaBoost BDT": get_stats(ada_model, X_test, Y_test),
        "Conifer Gradient BDT": get_stats(conif_model, X_test, Y_test), 
    }

    make_plot(results)

def plot_bitwidths():
    # Create a conifer config
    cfg = conifer.backends.vivadohls.auto_config()

    # Set the output directory to something unique
    cfg['OutputDir'] = 'tauTrees/prj_{}'.format(int(datetime.datetime.now().timestamp()))
    cfg['XilinxPart'] = 'xc7vx690tffg1761-2'
    cfg['Precision'] = 'ap_fixed<30, 15>'

    _, X, Y = load_data("../sig.csv", "../bg.csv", load_params_used("../params5_used.txt"))
    X_train, Y_train, X_test, Y_test = split_data(X, Y)
    model = open_model("../GradientBoosted_params5_trees50_depth2.pkl")

    conif_model = conifer.model(model, conifer.converters.sklearn, conifer.backends.vivadohls, cfg)
    conif_model.compile()
    results = { "Conifer Gradient BDT": get_stats(conif_model, X_test, Y_test)}
    make_plot(results)
    #conif_model.profile()
    #plt.show()

def main():
    ada_grad_conif_compare()
    # plot_bitwidths()

if __name__=="__main__":
    main() 
