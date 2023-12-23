from utils_functions.imports import *

# drawings_file = open("drawings_labels.txt","w+")
# images_file = open("images_labels.txt","w+")

drawings_labels_dict, images_labels_dict = {}, {}

nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

allowed_labels = ['D2', 'D21', 'D36', 'D4', 'D46', 'D58', 'E23', 'E34', 'F31', 'F35', 'G1', 'G17',
                  'G43', 'I10', 'I9', 'M17', 'M23', 'N35', 'O1', 'O34', 'O4', 'O49', 'Q1', 'Q3',
                  'R4', 'R8', 'S29', 'S34', 'U7', 'V13', 'V28', 'V30', 'V31', 'W11', 'W24', 'X1',
                  'X8', 'Y1', 'Y5', 'Z1']

for file in sorted(os.listdir("combined/drawings")):
    for i in range(1, len(file)):
        if file[i] not in nums:
            label = file[:i]
            break
    if label not in drawings_labels_dict.keys():
        drawings_labels_dict[label] = 1
    else:
        drawings_labels_dict[label]+=1
    if label in allowed_labels:
        if not os.path.isdir("combined_filtered/drawings/" + label):
            os.mkdir("combined_filtered/drawings/" + label)
        shutil.copy2("combined/drawings/" + file, "combined_filtered/drawings/" + label + "/" + file)


for file in os.listdir("combined/photos"):
    label = file.split("_")[1].split(".")[0]
    if label not in images_labels_dict.keys():
        images_labels_dict[label] = 1
    else:
        images_labels_dict[label]+=1
    if label in allowed_labels:
        if not os.path.isdir("combined_filtered/photos/" + label):
            os.mkdir("combined_filtered/photos/" + label)
        shutil.copy2("combined/photos/" + file, "combined_filtered/photos/" + label + "/" + file)

count_images, count_drawings = 0, 0

allowed_images_dict, allowed_drawings_dict = {}, {}

for label in allowed_labels:
    count_images+=images_labels_dict[label]
    count_drawings+=drawings_labels_dict[label]
    allowed_images_dict[label]=images_labels_dict[label]
    allowed_drawings_dict[label] = drawings_labels_dict[label]
    print("label {} images {} drawings {}".format(label, images_labels_dict[label], drawings_labels_dict[label]))

