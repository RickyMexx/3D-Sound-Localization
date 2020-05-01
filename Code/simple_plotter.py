import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import sample, randint, seed
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


def get_points_per_class(file_path):
    f = open(file_path, "r")

    #v is a list of (N, 11, 3) so every element is a point (3) for each class (11)
    v = []

    for i, linea in enumerate(f.readlines()):
        tokens = linea.strip("[ ]\n").split()
        
        riga = []
        for j in range(0, len(tokens), 3):
            riga.append([float(tokens[j][:-1]), float(tokens[j+1][:-1]), float(tokens[j+2][:-1])])
        v.append(riga)

    #classes is a list of (11, N, 3). Every class has it list of points
    classes = [[],[],[],[],[],[],[],[],[],[],[]]
    for line in v:
        for i,c in enumerate(line):
            classes[i].append(c)
    return classes


def plot_3d(doa_gt_file_path, doa_pred_file_path, from_class = 0, to_class = 11, rn_samples = 200, seed_ = 0):

    seed(seed_)

    gt_classes = get_points_per_class(doa_gt_file_path)
    pred_classes = get_points_per_class(doa_pred_file_path)

    idxs = sample(list(range(len(gt_classes[0]))), rn_samples)
    fig = plt.figure(figsize=(12.8, 4.8))
    final_plot_gt = []
    final_plot_pred = []

    classes = ['knock','drawer','clearthroat','phone','keysDrop','speech',
               'keyboard','pageturn','cough','doorslam','laughter']


    ax_pred = fig.add_subplot(121, projection='3d')
    ax_gt = fig.add_subplot(122, projection='3d')
    for gt_, pred_ in zip(gt_classes[from_class: to_class+1], pred_classes[from_class: to_class+1]):
        x_gt = []
        y_gt = []
        z_gt = []
        x_pred = []
        y_pred = []
        z_pred = []
        
        for idx in idxs:
            x_gt.append(gt_[idx][0])
            y_gt.append(gt_[idx][1])
            z_gt.append(gt_[idx][2])
            x_pred.append(round(pred_[idx][0], 1))
            y_pred.append(round(pred_[idx][1], 1))
            z_pred.append(round(pred_[idx][2], 1))
        
        final_plot_gt.append([x_gt,y_gt,z_gt])
        final_plot_pred.append([x_pred, y_pred, z_pred])

    for i,c in enumerate(final_plot_pred):
        ax_pred.scatter(c[0], c[1], c[2], marker = "o", label=classes[i+from_class])
    for i,c in enumerate(final_plot_gt):    
        ax_gt.scatter(c[0], c[1], c[2], marker = "o", label=classes[i+from_class])

    ax_pred.set_xlim(ax_gt.get_xlim())
    ax_pred.set_ylim(ax_gt.get_ylim())
    ax_pred.set_zlim(ax_gt.get_zlim())
    ax_gt.legend()
    ax_pred.legend()

    plt.savefig(plots_folder + '3d-plot.png')
    plt.clf()
    plt.cla()

    #plt.show()


def save_array_to_csv(file_name, array_to_save):
    f = open(file_name, "a")

    string_to_write = ""
    for elem in array_to_save:
        s = "%f," % (float(elem))
        string_to_write += s
    #Remove the last comma
    string_to_write = string_to_write[:-1]
    
    f.write(string_to_write+"\n")

    #(quite) fail proof, it's slower but in case of crashes the file is saved!
    f.close()


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

def plot_confidence(x, y, z, name):
    fig, ax = plt.subplots()
    ax.set_title('Confidence interval ' + name)
    ax.boxplot(x + y + z, 0, sym='')
    fig.savefig(plots_folder + 'confidence_int_' + name + '.png')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Method usage: python simple_plotter.py [file_name.csv]")
    
    csv_file = sys.argv[1]
    
    simple_plotter(csv_file, [0, 1], 'loss.png')
    simple_plotter(csv_file, [2, 3], 'er_f1.png')
    simple_plotter(csv_file, [10, 11, 12], 'scores.png')

    print("... Done! Plots can be found in /plots folder.")


