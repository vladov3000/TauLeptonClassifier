import conifer
import datetime
import pickle
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_plot(
    results, 
    colors=['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'],
    linewidth=2,
    plots_folder='./plots/plot.png'
    ):
    """
    Results is a dictionary where the key is the name of the model and the value is a
    dictionary with the keys fpr, tpr, auc. Plots these results with pretty colors.
    """
    plt.figure()
 
    c = 0
    for name, stats in results.items():
        plt.plot(stats["fpr"], stats["tpr"], color=colors[c], 
            lw=linewidth, label=f"{name} (area = {stats['auc']})")
        c = (c + 1) % len(colors)
 
    plt.semilogy()
    plt.xlim([0.0, 1.0])
    plt.ylim([1e-3, 1.0])
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    plt.title('ROC Curves for BDT Tau Lepton Classifier')
    plt.legend(loc="upper left")
    plt.savefig(plots_folder, bbox_inches='tight')
    plt.show()


def ada_grad_conif_compare():
    grad_model = open_model("../GradientBoosted_params5_trees50_depth2.pkl")
    ada_model = open_model("../params5_trees800_depth4.pkl")
    conif_model = synth_model(grad_model)

    _, X, Y = load_data("../sig.csv", "../bg.csv", load_params_used("../params5_used.txt"))
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

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
    # ada_grad_conif_compare()
    plot_bitwidths()

if __name__=="__main__":
    main() 
