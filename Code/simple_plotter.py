import matplotlib.pyplot as plt


file_name = "models/foa_train[3]_validation[2]_seq64_plot.csv"


def simple_plotter(file_name, column_number):
    f = open(file_name, "r")
    y = []
    
    for linea in f.readlines():
        elements = linea.split(",")
        y.append(float(elements[column_number]))
    
    x = range(len(y))

    return x,y


if __name__ == "__main__":
    x1, y1 = simple_plotter(file_name, 0)
    x2, y2 = simple_plotter(file_name, 1)
    x3, y3 = simple_plotter(file_name, 10)

    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.plot(x3, y3)
    plt.show()





