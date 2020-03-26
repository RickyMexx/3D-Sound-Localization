

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
