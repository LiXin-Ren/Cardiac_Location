import nibabel as nib
import os
import csv
train_dir = "Y:/gitFiles/cardiac_location_data/data"
label_dir = "Y:/gitFiles/cardiac_location_data/label"
trainCSV = "data.csv"
labelCSV = "label.csv"
def enumerateData(dir, csvfile):         #统计数据信息
    if not os.path.exists(dir):
        print("Dir not exists!")
        return 0
    with open(csvfile, "w", newline='') as csvFile:
        #csv_writer = csv.writer(csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)

        csvWriter = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(["name", "type", "id", "size"])
        for root, _, files in os.walk(dir):
            for fileName in files:
                niiData = nib.load(root + '/' + fileName)
                type = root.split('/')[-1]
                id = fileName.split('.')[0].split("_")[-1]
                size = niiData.shape
                csvWriter.writerow([fileName, type, id, size])




if __name__ == "__main__":
    # enumerateData(train_dir, trainCSV)
    enumerateData(label_dir, labelCSV)