import matplotlib.pyplot as plt
import sys

plots_folder = 'plots/'
labels = [  "Training loss", 
            "Validation loss", 
            "Sed ER", 
            "Sed F1", 
            "doa_loss_avg_accuracy", 
            "doa_loss_gt", 
            "doa_loss_pred", 
            "doa_loss_gt_cnt", 
            "doa_loss_pred_cnt", 
            "doa_loss_good_frame_cnt", 
            "Sed score", 
            "Doa score", 
            "Seld score", 
            "doa_confidence_interval_low", 
            "doa_confidence_interval_median", 
            "doa_confidence_interval_up", 
            "sed_confidence_interval_low", 
            "sed_confidence_interval_median", 
            "sed_confidence_interval_conf_up" ]


def simple_plotter(csv_file, columns, img_name):
    for c in columns:
        f = open(csv_file, "r")
        y = []
        for line in f.readlines():
            elements = line.split(",")
            y.append(float(elements[c]))
        f.close()
        x = range(len(y))
        plt.plot(x, y, label=labels[c])
    
    plt.legend()
    plt.savefig(plots_folder + img_name)
    plt.clf()
    plt.cla()
    return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Method usage: python simple_plotter.py [file_name.csv]")
    
    csv_file = sys.argv[1]
    
    simple_plotter(csv_file, [0, 1], 'loss.png')
    simple_plotter(csv_file, [2, 3], 'er_f1.png')
    simple_plotter(csv_file, [10, 11, 12], 'scores.png')

    print("... Done! Plots can be found in /plots folder.")


