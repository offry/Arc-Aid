import os
import math
from utils_functions.imports import *

drawings_labels_dict, images_labels_dict = {}, {}

nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

allowed_labels = ['D2', 'D21', 'D36', 'D4', 'D46', 'D58', 'E23', 'E34', 'F31', 'F35', 'G1', 'G17',
                  'G43', 'I10', 'I9', 'M17', 'M23', 'N35', 'O1', 'O34', 'O4', 'O49', 'Q1', 'Q3',
                  'R4', 'R8', 'S29', 'S34', 'U7', 'V13', 'V28', 'V30', 'V31', 'W11', 'W24', 'X1',
                  'X8', 'Y1', 'Y5', 'Z1']

lines, drawing_lines = [], []
for label in os.listdir("combined_filtered/photos"):
    label_dir = "combined_filtered/photos/" + label
    label_list = os.listdir(label_dir)
    test_idxs = list(random.sample(range(0, len(label_list)), math.ceil(len(label_list)*0.3)))
    drawing_label_dir = "combined_filtered/drawings/" + label
    drawing_label_list = os.listdir(drawing_label_dir)
    drawing_test_idx = list(random.sample(range(0, len(drawing_label_list)), len(drawing_label_list)))
    # drawing_train_idx = list(drawing_test_idx[1:])
    drawing_train_idx = list(drawing_test_idx)
    for i in range(0, len(label_list)):
        # if i in test_idxs:
        #     line = label_list[i] + "*" + label + "*" + "test\n"
        #     drawing_line = drawing_label_list[drawing_test_idx[0]] + "*" + label + "*" + "test\n"
        if len(test_idxs)==1:
            line = label_list[i] + "*" + label + "*" + "val\n"
            drawing_line = drawing_label_list[drawing_test_idx[0]] + "*" + label + "*" + "val\n"
            continue
        if i in test_idxs[int(len(test_idxs)/2):]:
            line = label_list[i] + "*" + label + "*" + "test\n"
            drawing_line = drawing_label_list[drawing_test_idx[0]] + "*" + label + "*" + "test\n"
        elif i in test_idxs[:int(len(test_idxs) / 2)]:
            line = label_list[i] + "*" + label + "*" + "val\n"
            drawing_line = drawing_label_list[drawing_test_idx[0]] + "*" + label + "*" + "val\n"
        else:
            line = label_list[i] + "*" + label + "*" + "train\n"
            drawing_line = drawing_label_list[drawing_train_idx[i%len(drawing_train_idx)]] + "*" + label + "*" + "train\n"
        # drawing_line = drawing_label_list[drawing_train_idx[i % len(drawing_train_idx)]] + "*" + label + "*" + "train\n"
        lines.append(line)
        drawing_lines.append(drawing_line)

drawing_split_file = open('combined_filtered/original_experiment/drawings/split.txt', 'w')
drawing_split_file.writelines(drawing_lines)
drawing_split_file.close()
split_file = open('combined_filtered/original_experiment/photos/split.txt', 'w')
split_file.writelines(lines)
split_file.close()


# drawing_lines = []
# for label in os.listdir("combined_filtered/drawings"):
#     drawing_label_dir = "combined_filtered/drawings/" + label
#     drawing_label_list = os.listdir(drawing_label_dir)
#     drawing_test_idx = list(random.sample(range(0, len(drawing_label_list)), len(drawing_label_list)))
#     drawing_train_idx = list(drawing_test_idx[1:])
#     for i in range(0, len(drawing_label_list)):
#         if i in drawing_test_idx[:1]:
#             drawing_line = drawing_label_list[i] + "*" + label + "*" + "test\n"
#         else:
#             drawing_line = drawing_label_list[i] + "*" + label + "*" + "train\n"
#         drawing_lines.append(drawing_line)
#
# drawing_split_file = open('combined_filtered/experiment/drawings/split.txt', 'w')
# drawing_split_file.writelines(drawing_lines)
# drawing_split_file.close()

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from io import BytesIO
from cairosvg import svg2png
# for label in os.listdir("combined_filtered/drawings/"):
#     list_dir = os.listdir("combined_filtered/drawings/" + label)
#     for d in list_dir:
#         if "svg" in d:
#             # drawing = svg2rlg("combined_filtered/drawings/" + label + "/" + d)
#             # renderPM.drawToFile(drawing, "combined_filtered/drawings/" + label + "/" + d.split(".")[0] + ".png", fmt='PNG')
#             png = svg2png(url="combined_filtered/drawings/" + label + "/" + d, output_width=224, output_height=224)
#             pil_img = Image.open(BytesIO(png)).convert('RGBA')
#             new_image = Image.new("RGBA", pil_img.size, "WHITE")
#             new_image.paste(pil_img, (0, 0),
#                             pil_img)  # Paste the image on the background. Go to the links given below for details.
#             new_image.convert('RGB').save("combined_filtered/drawings/" + label + "/" + d.split(".")[0] + ".tiff")
#
#             os.remove("combined_filtered/drawings/" + label + "/" + d,)
#         elif "png" in d:
#             image = Image.open("combined_filtered/drawings/" + label + "/" + d)
#             new_image = Image.new("RGBA", image.size, "WHITE")  # Create a white rgba background
#             new_image.paste(image, (0, 0),
#                             image)  # Paste the image on the background. Go to the links given below for details.
#             new_image.convert('RGB').save("combined_filtered/drawings/" + label + "/" + d.split(".")[0] + ".tiff")
#             os.remove("combined_filtered/drawings/" + label + "/" + d, )
#
# list_dir = os.listdir("combined/drawings/1")
# for d in list_dir:
#     if "svg" in d:
#         # drawing = svg2rlg("combined_filtered/drawings/" + label + "/" + d)
#         # renderPM.drawToFile(drawing, "combined_filtered/drawings/" + label + "/" + d.split(".")[0] + ".png", fmt='PNG')
#         png = svg2png(url="combined/drawings/1" + "/" + d, output_width=224, output_height=224)
#         pil_img = Image.open(BytesIO(png)).convert('RGBA')
#         pil_img.save("combined/drawings/1" + "/" + d.split(".")[0] + ".tiff")
#
#         os.remove("combined/drawings/1" + "/" + d, )
#     elif "png" in d:
#         image = Image.open("combined/drawings/1" + "/" + d)
#         new_image = Image.new("RGBA", image.size, "WHITE")  # Create a white rgba background
#         new_image.paste(image, (0, 0),
#                         image)  # Paste the image on the background. Go to the links given below for details.
#         new_image.convert('RGB').save("combined/drawings/1" + "/" + d.split(".")[0] + ".tiff")
#         os.remove("combined/drawings/1" + "/" + d, )