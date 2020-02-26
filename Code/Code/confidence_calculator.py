import numpy as np
import os
import scipy.stats as st


METADATA_FOLDER = "../Dataset/metadata_dev"


def read_csv(az, el):
    for file in os.listdir(METADATA_FOLDER):
        file_name = os.path.join(METADATA_FOLDER, file)

        f = open(file_name).readlines()[1:]

        for n, riga in enumerate(f):
            el.append(float(riga.split(",")[3]))
            az.append(float(riga.split(",")[4]))

def compute_confidence_interval(data, confidence=0.95):
    card = len(data) # Cardinality of the dataset
    mean = np.mean(data) # Mean value of data
    sem = st.sem(data) # Standard error of the mean
    # h = sem * scipy.stats.f.ppf((1+confidence)/2, card-1) # Computing h with Percent point function (inverse of cdf, cumulative density function).
    # return [mean + h, mean - h]
    return st.t.interval(confidence, card - 1, loc=mean, scale=sem)
        
    

if __name__ == "__main__":
    az = []
    el = []

    read_csv(az, el)

    print(len(az))
    print(len(el))

    # Computing confidence intervals
    [az_s, az_e] = compute_confidence_interval(az)
    [el_s, el_e] = compute_confidence_interval(el)

    print("\nConfidence interval of azimuth:")
    print("Mean value:"+str(np.mean(az)))
    print("Displacement h: +/- "+str(np.abs(np.mean(az)-az_s)))
    print("["+str(az_s) + ", " + str(az_e)+"]")

    print("\nConfidence interval of elevation:")
    print("Mean value:" + str(np.mean(el)))
    print("Displacement h: +/- " + str(np.abs(np.mean(el) - el_s)))
    print("["+str(el_s) + ", " + str(el_e)+"]")

