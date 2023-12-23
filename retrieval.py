import argparse
import os

from utils_functions.imports import *
from utils_functions.util_functions import *
from util_models.util_models import *
from utils_functions.dataloaders_and_augmentations import *
from supervised_training.train_supervised import *
from utils_functions.data_dirs_functions import *
from operator import itemgetter
import kornia.losses
import pytorch_metric_learning.losses
import torch.nn

from utils_functions.imports import *
from util_models.util_models import *
from utils_functions.util_functions import *
from utils_functions.util_train_functions import *
from utils_functions.util_train_functions import *
from util_models.resnet_with_skip import Resnet_with_skip
from util_models.densenet_with_skip import Densenet_with_skip
from util_models.glyphnet_with_skip import Glyphnet_with_skip
from utils_functions.dataloaders_and_augmentations import *
from util_models.efficientnet_with_skip import *
from util_models.coinnet_with_skip import *


class embedding_coinnet(nn.Module):
    def __init__(self, coinnet):
        super(embedding_coinnet, self).__init__()

        self.model_d161 = coinnet.module.model_d161


    def forward(self, x):
        _, d161_features, _ = self.model_d161(x)

        return d161_features


class embedding_glyphnet(nn.Module):
    def __init__(self, glyphnet):
        super(embedding_glyphnet, self).__init__()

        self.first_block = glyphnet.first_block
        self.inner_blocks = glyphnet.inner_blocks
        self.final_block = glyphnet.final_block


    def forward(self, x):
        x = self.first_block(x)
        x = self.inner_blocks(x)
        x = F.relu(self.final_block.bn(self.final_block.sconv(x)))
        x = torch.mean(x, dim=(-1, -2))
        return x


def give_label_score(query_label, retrieved_label):
    if query_label == retrieved_label:
        return 1
    else:
        return 0


def knn_calc(image_name, query_feature, features):
    current_image_feature = features[image_name]
    criterion = torch.nn.CosineSimilarity(dim=1)
    dist = criterion(query_feature, current_image_feature).mean()
    dist = -dist.item()
    return dist


def create_figure(query_image_path, topk_labels, res, original_query_label, k, labels_dict):
    query_image = cv2.imread(query_image_path)
    query_image_resized = cv2.resize(query_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    fig = plt.figure(figsize=[k * 3, k * 3])
    cols = int(k / 2)
    rows = int(k / cols) + 1
    ax_query = fig.add_subplot(rows, 1, 1)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.imshow(cv2.cvtColor(query_image_resized, cv2.COLOR_BGR2RGB))
    ax_query.set_title('Query image is {}'.format(original_query_label), fontsize=40)
    for i, image in zip(range(len(res)), res):
        current_image_path = image
        original_label = check_image_label(current_image_path, labels_dict)
        if i < k:
            image = cv2.imread(current_image_path)
            image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            ax = fig.add_subplot(rows, cols, i + 6)
            plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
            ax.set_title('Top {}, image is {}'.format(i + 1, original_label), fontsize=40)
        topk_labels.append(original_label)
    return topk_labels, fig


def return_all_features(model_test, query_images_paths):
    model_test.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    features = dict()
    i = 0
    transform = transforms.Compose([
            transforms.RandomApply([transforms.ToPILImage(),], p=1),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    with torch.no_grad():
        for image_path in query_images_paths:
            i = i + 1
            # if check_image_label(image_path, labels_dict) is not None:
            img = cv2.imread(image_path)
            img = transform(img)
            # img = transforms.Grayscale(num_output_channels=1)(img).to(device)
            img = img.unsqueeze(0).contiguous().to(device)
            current_image_features = model_test(img)
            # current_image_features, _, _, _ = model_test(x1=img, x2=img)
            features[image_path] = current_image_features
            # if i % 5 == 0:
            #     print("Finished embedding of {} images".format(i))
            del current_image_features
            torch.cuda.empty_cache()
    return features


def check_retrieval(checkpoint_path, args, training_list, exp):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arch_type = args.arch_type
    training_proceedure = training_list[0]
    model_file = checkpoint_path
    checkpoint = torch.load(checkpoint_path + args.arch_type + "_" + exp +
                            "/" + exp + "_best_test_accuracy.pth.tar", map_location="cpu")
    model_test = get_model(arch_type, args.num_classes, args).to(device)
    model_test.load_state_dict(checkpoint)
    if "resnet" in arch_type:
        embedding_model_test = torch.nn.Sequential(*(list(model_test.children())[:-1]))
    elif "efficientnet" in arch_type:
        embedding_model_test = torch.nn.Sequential(*(list(model_test.children())[:-2]))
    elif "densenet" in arch_type:
        embedding_model_test = torch.nn.Sequential(*(list(model_test.children())[:-2]))
    elif "glyphnet" in arch_type:
        embedding_model_test = embedding_glyphnet(model_test)
    elif "coinnet" in arch_type:
        embedding_model_test = embedding_coinnet(model_test)
    data_dir = "queries/" + str(exp)
    query_images_paths = []
    for path in os.listdir(data_dir):
        query_images_paths.append(os.path.join(data_dir, path))
    features = return_all_features(embedding_model_test, query_images_paths)
    for query_image_name in query_images_paths:
        dists = dict()
        query_feature = features[query_image_name]
        with torch.no_grad():
            for i, image_name in enumerate(query_images_paths):
                if image_name == query_image_name:
                    continue
                dist = knn_calc(image_name, query_feature, features)
                dists[image_name] = dist
        res = dict(sorted(dists.items(), key=itemgetter(1)))
        if not os.path.isdir("queries/results"):
            os.mkdir("queries/results")
        if not os.path.isdir("queries/results/" + arch_type + "_" + training_proceedure + "_" + str(exp)):
            os.mkdir("queries/results/" + arch_type + "_" + training_proceedure + "_" + str(exp))
        exp_file = open("queries/results/" + arch_type + "_" + training_proceedure + "_" + str(exp)+"/"+
                        query_image_name.split("/")[2].split(".tif")[0]+".txt", "w")
        exp_file.writelines(s + '\n' for s in res)
        exp_file.close()
        print(query_image_name)


def retrieval_calc_map(args, checkpoint_path, training_list, exp):
    check_retrieval(checkpoint_path, args, training_list, exp)
    average_map = 0.0
    arch_type = args.arch_type
    training_proceedure = training_list[0]
    data_dir = "queries/" + str(exp)
    query_images_paths = []
    label_dict = {}
    for path in os.listdir(data_dir):
        query_images_paths.append(os.path.join(data_dir, path))
        found = False
        for label in os.listdir("cssl_dataset/shape/photos"):
            for name in os.listdir("cssl_dataset/shape/photos/" + label):
                if name == path:
                    label_dict[os.path.join(data_dir, path)] = label
                    found = True
                    break
            if found:
                break
    k = 10
    map_score = 0.0
    for query_image_name in query_images_paths:
        exp_file = open("queries/results/" + arch_type + "_" + training_proceedure + "_" + str(exp)+"/"+
                        query_image_name.split("/")[2].split(".tif")[0]+".txt", "r")
        res = exp_file.readlines()
        ap_score = 0.0
        query_label = label_dict[query_image_name]
        for current_k in range(1, k + 1):
            same_label_counter = 0
            current_p_at_k = 0.0
            for i, data_path in zip(range(1, current_k + 1), res[:current_k]):
                data_label = label_dict[data_path[:-1]]
                current_query_label_score = 0.0
                if query_label is not None and data_label is not None:
                    current_query_label_score = give_label_score(query_label, data_label)
                    if current_query_label_score == 1:
                        same_label_counter = same_label_counter + 1
                current_p_at_k += current_query_label_score
            current_p_at_k = current_p_at_k / current_k
            ap_score = ap_score + current_p_at_k
        ap_score = ap_score/k
        map_score +=ap_score
    map_score = map_score / len(query_images_paths)
    average_map += map_score
    print(" {} map score for {} {} with P@{} is: {}".format(exp, training_list[0], arch_type, k, map_score))