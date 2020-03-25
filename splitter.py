import os

dir_name = "."

for file_name in os.listdir(dir_name):
    if("test_" in file_name):
	    new_name = file_name[5:]
	    print(new_name)
	    os.rename(file_name, new_name)
    if("train_" in file_name):
        new_name = file_name[6:]
        print(new_name)
        os.rename(file_name, new_name)
