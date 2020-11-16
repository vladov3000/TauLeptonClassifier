import conifer
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
    cfg['OutputDir'] = 'tauTrees/prj_1605048046/rj_{}'.format(int(datetime.datetime.now().timestamp()))
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

def get_stats(model, X, Y):
    """ 
    Returns false positive rates, true positive rates, 
    and area under curve for given model. 
    """
    Y_pred = model.decision_function(X)
    fpr, tpr, _ = sklearn.metrics.roc_curve(Y, Y_pred)
    auc = sklearn.metrics.auc(fpr, tpr) 
    return dict(zip(["fpr", "tpr", "auc"], [fpr, tpr, auc]))

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


def main():
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

if __name__=="__main__":
    main() 
