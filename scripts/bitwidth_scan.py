from utils import *

def get_auc_ratios(grad_model, X_test, Y_test, int_bit, end_bit):
    sklearn_stats = get_stats(grad_model, X_test, Y_test) 

    bit_widths = []
    auc_ratios = []
    aucs = []

    for i in range(int_bit, end_bit):
        bw = 'ap_fixed<{},{}>'.format(i, int_bit)
        print(f'calculating {bw}...')
        bit_widths.append(bw)
        conif_model = synth_model(grad_model, bit_width=bw, build=False)
        conif_stats = get_stats(conif_model, X_test, Y_test)
        auc_ratios.append(conif_stats["auc"]/sklearn_stats["auc"])
        aucs.append(conif_stats["auc"])

    return bit_widths, auc_ratios, aucs

def plot_auc_ratios(auc_ratios, bit_widths):
    plt.cla()
    plt.scatter(bit_widths, auc_ratios)
    plt.title('Conifer Model Bit Width Profile')
    plt.xlabel('Fixed Point Bit Width <Total Bit Width, Integer Width>')
    plt.ylabel('Conifer AUC/Sklearn AUC')
    plt.xticks(rotation=45)
    plt.savefig(f'../images/{bit_widths[0]}-{bit_widths[-1]}.png')

def main():
    X_train, Y_train, X_test, Y_test = load_split_data()
    grad_model = open_model('../models/scaled_100_4_model.pkl')
    for j in range(5, 19):
        print(f'calculating {j} ...')
        bit_widths, auc_ratios, aucs = get_auc_ratios(grad_model, X_test, Y_test, j, 19)
        print(f'plotting {j} ...')
        plot_auc_ratios(auc_ratios, bit_widths)

if __name__ == "__main__":
    main()
