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
    results = { f"Gradient BDT {cfg['Precision']}": get_stats(conif_model, X_test, Y_test) }
    return results

def main():
    plot_bitwidths()

if __name__=="__main__":
    main()
