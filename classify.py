import torch
import os
from utils_functions.imports import *


def eval_classification(arch_type, checkpoint_path, args, device, dataloaders_test, dataset_sizes, experiment):
    model = get_model(arch_type, 10, args)
    checkpoint = torch.load(checkpoint_path + arch_type + "_" + experiment +
                            "/" + experiment + "_best_test_accuracy.pth.tar", map_location="cpu")
    model.to(device)
    model.load_state_dict(checkpoint)
    corrects = 0
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    gray_scale = transforms.Grayscale(num_output_channels=1)
    for inputs, labels, _ in dataloaders_test:
        inputs = inputs.to(device)
        if isinstance(labels, list):
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels)

        labels = labels.to(device)
        with torch.no_grad():
            if "glyphnet" in arch_type:
                outputs = model(gray_scale(inputs))
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
    acc = 100 * corrects.float() / dataset_sizes['test']
    print("Classification accuracy of {} on {} is {}".format(checkpoint_path, experiment, acc))