from utils_functions.imports import *
from utils_functions.dataloaders_and_augmentations import *
import wandb
import torch
import os


def wandb_init(args,
                                   training_procedure, exp, experiment_name, experiment_procedure, project_name):
    run_wandb = wandb.init(project=project_name, allow_val_change=True, entity="offry", reinit=True)
    run_wandb.name = experiment_name
    run_wandb.save()
    run_wandb.config.training_procedure = training_procedure
    run_wandb.config.experiment_procedure = experiment_procedure
    run_wandb.config.update(args)
    return run_wandb


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def print_torch():
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))


def init_experiments(args):
    data_transforms = data_transforms_arch(args.image_size, args)
    if args.shapes_classification:
        project_name = "publish_classification_by_shape"
        args.num_classes = 10
    if args.periods_classification:
        if args.sub_periods:
            args.num_classes = 5
            project_name = "publish_classification_by_sub_periods"
        else:
            args.num_classes = 3
            project_name = "publish_classification_by_periods"
    exp_list = []
    for exp_num in range(0, args.num_of_exps_to_eval):
        exp_list.append('experiment_' + str(exp_num))

    return args, exp_list, project_name, data_transforms


def create_experiment_name(training_procedure, args, exp, total):
    if args.periods_classification:
        if args.sub_periods:
            sub = "_sub"
        else:
            sub = ""
        if args.train_periods_with_shapes_network:
            experiment_procedure = training_procedure + sub + "_periods_with_shapes_" + args.shapes_net_type[:-11]
            experiment_procedure = experiment_procedure + "batch_" + str(args.batch_size)
        else:
            experiment_procedure = training_procedure + sub + "_batch_" + str(args.batch_size)
    else:
        experiment_procedure = training_procedure + "_batch_" + str(args.batch_size)
    if args.part_train_set:
        experiment_procedure = experiment_procedure + "part_train_set" + str(args.original_partial_train_set_ratio)
    if total:
        experiment_name = args.arch_type + "_" + "total" + "_" + experiment_procedure
    else:
        experiment_name = args.arch_type + "_" + exp + "_" + experiment_procedure
    return experiment_procedure, experiment_name


def create_checkpoints_dirs(args, exp, experiment_name):
    if args.shapes_classification:
        root = "shapes"
        multi = "regular_shape/"
    if args.periods_classification:
        if args.sub_periods:
            root = "sub_periods_with_shapes"
        else:
            root = "periods_with_shapes"
        multi = ""
    if args.part_train_set:
        if args.original_partial_train_set_ratio==2.0:
            image_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + multi + root + "/train_sets/33-67/photos/" + exp
            draw_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + multi + root + "/train_sets/33-67/drawings/" + exp
        elif args.original_partial_train_set_ratio==4.0:
            image_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + multi + root + "/train_sets/50-50/photos/" + exp
            draw_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + multi + root + "/train_sets/50-50/drawings/" + exp
        elif args.original_partial_train_set_ratio ==8.0:
            image_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + multi + root + "/train_sets/67-33/photos/" + exp
            draw_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + multi + root + "/train_sets/67-33/drawings/" + exp
        elif args.original_partial_train_set_ratio==16.0:
            image_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + multi + root + "/train_sets/80-20/photos/" + exp
            draw_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + multi + root + "/train_sets/80-20/drawings/" + exp
    else:
        image_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + multi + root + "/train_sets/20-80/photos/" + exp
        draw_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + multi + root + "/train_sets/20-80/drawings/" + exp
    if args.shapes_classification:
        task_dir = "shapes_classification"
    if args.periods_classification:
        if args.train_periods_with_shapes_network:
            base_data_dir = "periods_with_shapes_" + args.shapes_net_type[:-11]
        else:
            base_data_dir = "periods_with_shapes"
        if args.sub_periods:
            base_data_dir = "sub_" + base_data_dir
        task_dir = "periods_classification/" + base_data_dir
    draw_exp_name = experiment_name.split(exp)[0] + exp + "_train_drawings_batch_" + str(args.batch_size)
    draw_checkpoint_dir = os.path.join(os.getcwd(),
                                  task_dir + "/checkpoints/" + draw_exp_name + "_" + str(
                                        args.num_classes) + "_classes")
    image_checkpoint_dir = os.path.join(os.getcwd(),
                                        task_dir + "/checkpoints/checkpoints_photo_" + args.arch_type + "_" + exp + "_num_classes_" + str(
                                            args.num_classes))
    return image_dir, draw_dir, draw_checkpoint_dir, image_checkpoint_dir


def create_checkpoints_dirs_for_shapes_and_periods(args, exp, experiment_name):

    image_dirs_dict, draw_dirs_dict = {}, {}
    if args.train_periods_with_shapes_network:
        base_data_dir = "periods_with_shapes"
        root_list = [""]
    else:
        base_data_dir = "periods_with_shapes"
        root_list = [""]
    if args.sub_periods:
        base_data_dir = "sub_" + base_data_dir

    for root in root_list:
        if args.part_train_set:
            if args.original_partial_train_set_ratio==2.0:
                image_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + base_data_dir + root + "/train_sets/33-67/photos/" + exp
                draw_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + base_data_dir + root + "/train_sets/33-67/drawings/" + exp
            elif args.original_partial_train_set_ratio==4.0:
                image_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + base_data_dir + root + "/train_sets/50-50/photos/" + exp
                draw_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + base_data_dir + root + "/train_sets/50-50/drawings/" + exp
            elif args.original_partial_train_set_ratio ==8.0:
                image_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + base_data_dir + root + "/train_sets/67-33/photos/" + exp
                draw_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + base_data_dir + root + "/train_sets/67-33/drawings/" + exp
            elif args.original_partial_train_set_ratio==16.0:
                image_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + base_data_dir + root + "/train_sets/80-20/photos/" + exp
                draw_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + base_data_dir + root + "/train_sets/80-20/drawings/" + exp
        else:
            image_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + base_data_dir + root + "/train_sets/20-80/photos/" + exp
            draw_dir = os.path.join(args.cssl_dataset_bast_dir, "experiments/") + base_data_dir + root + "/train_sets/20-80/drawings/" + exp
        if root!="":
            root = root[1:]
        image_dirs_dict[root] = image_dir
        draw_dirs_dict[root] = draw_dir

    task_dir = "periods_classification/" + base_data_dir
    if args.train_periods_with_shapes_network:
        task_dir += "_with_shapes_" + args.shapes_net_type[:-11]
    else:
        task_dir += "_with_shapes"

    image_checkpoint_dir = os.path.join(os.getcwd(),
                                        task_dir + "/checkpoints/checkpoints_photo_" + args.arch_type + "_" + exp + "_num_classes_" + str(
                                            args.num_classes))
    return image_dirs_dict, draw_dirs_dict, image_checkpoint_dir


def init_plot_predictions(class_names):
    incorrect_examples, original_labels, pred_labels, paths_of_incorrect, real_class_names = [], [], [], [] ,[]
    real_classes_num, running_corrects = 0, 0
    for sub_class in class_names:
        if "_" in sub_class:
            if sub_class.split("_")[0] in real_class_names:
                continue
            else:
                real_class_names.append(sub_class.split("_")[0])
                real_classes_num += 1
        else:
            if sub_class.split("_")[0] in real_class_names:
                continue
            else:
                real_class_names.append(sub_class)
                real_classes_num += 1
    return incorrect_examples, original_labels, pred_labels, paths_of_incorrect, real_class_names


def final_eval_model(model_test, dataloaders_test, running_corrects, training_procedure, device, class_names,
                     incorrect_examples, original_labels, pred_labels, paths_of_incorrect, confusion_matrix, api_net=False):
    with torch.no_grad():
        if training_procedure == "train_photos" or training_procedure == "train_drawings":
            for inputs, labels, paths in dataloaders_test:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if api_net:
                    outputs = model_test(inputs, targets=None, flag='val')
                else:
                    outputs = model_test(inputs)
                _, preds = torch.max(outputs, 1)
                new_preds = preds
                for i in range(len(preds)):
                    pred = preds[i]
                    pred_name = class_names[preds[i]]
                    label_name = class_names[labels[i].item()]
                    if "_" in label_name:
                        label_name = label_name.split("_")[0]
                    if "_" in pred_name:
                        pred_name = pred_name.split("_")[0]
                    if label_name == pred_name:
                        new_preds[i] = labels[i]
                running_corrects += torch.sum(preds == labels.data)
                idxs_mask = (preds != labels).view(-1)
                incorrect_examples.append(inputs[idxs_mask].cpu())
                for idx, idx_bool in enumerate(idxs_mask):
                    if idx_bool:
                        paths_of_incorrect.append(paths[idx])
                original_labels.append(labels[idxs_mask].cpu())
                pred_labels.append(preds[idxs_mask].cpu())
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        else:
            for draw_data, image_data in dataloaders_test:
                draw_inputs, draw_labels, _ = draw_data
                image_inputs, image_labels, paths = image_data

                draw_inputs = draw_inputs.to(device)
                # draw_labels = draw_labels.to(device)
                image_inputs = image_inputs.to(device)
                image_labels = image_labels.to(device)

                outputs = model_test(image_inputs)

                _, preds = torch.max(outputs, 1)
                new_preds = preds
                for i in range(len(preds)):
                    pred = preds[i]
                    pred_name = class_names[preds[i]]
                    label_name = class_names[image_labels[i].item()]
                    if "_" in label_name:
                        label_name = label_name.split("_")[0]
                    if "_" in pred_name:
                        pred_name = pred_name.split("_")[0]
                    if label_name == pred_name:
                        new_preds[i] = image_labels[i]
                running_corrects += torch.sum(new_preds == image_labels.data)
                idxs_mask = (preds != image_labels).view(-1)
                incorrect_examples.append(image_inputs[idxs_mask].cpu())
                for idx, idx_bool in enumerate(idxs_mask):
                    if idx_bool:
                        paths_of_incorrect.append(paths[idx])
                original_labels.append(image_labels[idxs_mask].cpu())
                pred_labels.append(preds[idxs_mask].cpu())
                for t, p in zip(image_labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix, running_corrects, incorrect_examples, paths_of_incorrect, pred_labels, original_labels


def final_eval_model_shapes_with_periods(models_dict, model_test, dataloaders_test, running_corrects, training_procedure, device, class_names,
                     incorrect_examples, original_labels, pred_labels, paths_of_incorrect, confusion_matrix, args, final_classifier):
    with torch.no_grad():
        if training_procedure == "train_photos" or training_procedure == "train_drawings":
            for shapes_draw_data, shapes_image_data, base_draw_data, base_image_data, back_draw_data, back_image_data, \
                side_draw_data, side_image_data in dataloaders_test:
                output = torch.zeros(8, 2048, 1, 1).to(device)
                if args.train_periods_from_back and args.periods_classification:
                    draw_inputs, draw_labels, draw_paths = back_draw_data
                    image_inputs, image_labels, image_paths = back_image_data
                    if "drawings" in training_procedure:
                        inputs, labels, paths = draw_inputs, draw_labels, draw_paths
                    else:
                        inputs, labels, paths = image_inputs, image_labels, image_paths
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    model = models_dict["back"]
                    embedding_model = torch.nn.Sequential(*(list(model.children())[:-1]))
                    output += embedding_model(inputs)
                if args.train_periods_from_side and args.periods_classification:
                    draw_inputs, draw_labels, draw_paths = side_draw_data
                    image_inputs, image_labels, image_paths = side_image_data
                    if "drawings" in training_procedure:
                        inputs, labels, paths = draw_inputs, draw_labels, draw_paths
                    else:
                        inputs, labels, paths = image_inputs, image_labels, image_paths
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    model = models_dict["side"]
                    embedding_model = torch.nn.Sequential(*(list(model.children())[:-1]))
                    output += embedding_model(inputs)
                if args.train_periods_with_shapes_network and args.periods_classification:
                    draw_inputs, draw_labels, draw_paths = shapes_draw_data
                    image_inputs, image_labels, image_paths = shapes_image_data
                    if "drawings" in training_procedure:
                        shape_inputs, shape_labels, shape_paths = draw_inputs, draw_labels, draw_paths
                    else:
                        shape_inputs, shape_labels, shape_paths = image_inputs, image_labels, image_paths
                    shape_inputs = shape_inputs.to(device)
                    shape_labels = shape_labels.to(device)
                    model = models_dict["shapes"]
                    embedding_model = torch.nn.Sequential(*(list(model.children())[:-1]))
                    output = embedding_model(shape_inputs)
                if args.train_periods_from_base and args.periods_classification:
                    draw_inputs, draw_labels, draw_paths = base_draw_data
                    image_inputs, image_labels, image_paths = base_image_data
                    if "drawings" in training_procedure:
                        inputs, labels, paths = draw_inputs, draw_labels, draw_paths
                    else:
                        inputs, labels, paths = image_inputs, image_labels, image_paths
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    model = models_dict["base"]
                    embedding_model = torch.nn.Sequential(*(list(model.children())[:-1]))
                    output += embedding_model(inputs)

                output = torch.squeeze(output, 2)
                output = torch.squeeze(output, 2)
                outputs = model_test(output)
                _, preds = torch.max(outputs, 1)
                new_preds = preds
                if final_classifier == "final_shapes_base_classification_shapes":
                    labels = shape_labels
                    paths = shape_paths
                for i in range(len(preds)):
                    pred = preds[i]
                    pred_name = class_names[preds[i]]

                    label_name = class_names[labels[i].item()]
                    if "_" in label_name:
                        label_name = label_name.split("_")[0]
                    if "_" in pred_name:
                        pred_name = pred_name.split("_")[0]
                    if label_name == pred_name:
                        new_preds[i] = labels[i]
                running_corrects += torch.sum(preds == labels.data)
                idxs_mask = (preds != labels).view(-1)
                incorrect_examples.append(inputs[idxs_mask].cpu())
                for idx, idx_bool in enumerate(idxs_mask):
                    if idx_bool:
                        paths_of_incorrect.append(paths[idx])
                original_labels.append(labels[idxs_mask].cpu())
                pred_labels.append(preds[idxs_mask].cpu())
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        else:
            for draw_data, image_data in dataloaders_test:
                draw_inputs, draw_labels, _ = draw_data
                image_inputs, image_labels, paths = image_data

                draw_inputs = draw_inputs.to(device)
                # draw_labels = draw_labels.to(device)
                image_inputs = image_inputs.to(device)
                image_labels = image_labels.to(device)

                outputs = model_test(image_inputs)

                _, preds = torch.max(outputs, 1)
                new_preds = preds
                for i in range(len(preds)):
                    pred = preds[i]
                    pred_name = class_names[preds[i]]
                    label_name = class_names[image_labels[i].item()]
                    if "_" in label_name:
                        label_name = label_name.split("_")[0]
                    if "_" in pred_name:
                        pred_name = pred_name.split("_")[0]
                    if label_name == pred_name:
                        new_preds[i] = image_labels[i]
                running_corrects += torch.sum(new_preds == image_labels.data)
                idxs_mask = (preds != image_labels).view(-1)
                incorrect_examples.append(image_inputs[idxs_mask].cpu())
                for idx, idx_bool in enumerate(idxs_mask):
                    if idx_bool:
                        paths_of_incorrect.append(paths[idx])
                original_labels.append(image_labels[idxs_mask].cpu())
                pred_labels.append(preds[idxs_mask].cpu())
                for t, p in zip(image_labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix, running_corrects, incorrect_examples, paths_of_incorrect, pred_labels, original_labels


def plot_confusion_matrix(confusion_matrix, args, class_names,
                          training_procedure, exp, experiment_procedure, acc, total=False):
    fig = plt.figure(figsize=(22, 18))

    # plt.interactive(True)
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(float)
    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right', fontsize=15)
    if total:
        plt.title("Total {}, acc is {}".format(training_procedure, acc), fontsize=15)
    else:
        plt.title("{}, {}, acc is {}".format(exp, training_procedure, acc), fontsize=15)
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    # plt.show()
    if args.shapes_classification:
        task_dir = "shapes_classification"
    if args.periods_classification:
        task_dir = "periods_classification/"
        if args.sub_periods:
            task_dir+="sub_periods"
        else:
            task_dir+="periods"
        if args.train_periods_with_shapes_network:
            task_dir += "_with_shapes_" + args.shapes_net_type[:-11]
        else:
            task_dir += "_with_shapes"
        if not os.path.isdir(task_dir+"/results/figures/"):
            os.mkdir(task_dir+"/results/figures/")

    if not os.path.isdir(task_dir + "/results/figures/" + args.arch_type + "_" + experiment_procedure):
        os.mkdir(task_dir + "/results/figures/" + args.arch_type + "_" + experiment_procedure)
    fig.savefig(task_dir + "/results/figures/" + args.arch_type + "_" + experiment_procedure + "/" + exp + ".png")
    # run_wandb.log({exp + "_" + training_procedure: fig})
    plt.close


def plot_incorrect_examples(best_experiment, incorrect_examples, paths_of_incorrect, class_names, original_labels, pred_labels,
                            experiment_name, args, training_procedure):
    image_num = 0
    if args.shapes_classification:
        task_dir = "shapes_classification"
    if args.periods_classification:
        task_dir = "periods_classification/"
        if args.sub_periods:
            task_dir += "sub_periods"
        else:
            task_dir += "periods"

        if args.train_periods_with_shapes_network:
            task_dir += "_with_shapes_" + args.shapes_net_type[:-11]
        else:
            task_dir += "_with_shapes"

        if not os.path.isdir(task_dir + "/results/false_results/"):
            os.mkdir(task_dir + "/results/false_results/")
    false_dir = task_dir + "/results/false_results/" + experiment_name

    for j, item in enumerate(incorrect_examples):
        for i, image in enumerate(item):
            invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                                           transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                                std=[1., 1., 1.]),
                                           ])
            inv_tensor = invTrans(image)
            fig = plt.figure()
            image_name = paths_of_incorrect[image_num].split("/")[-1]
            plt.title("label: {} pred label: {}".format(class_names[int(original_labels[j][i])],
                                                        class_names[int(pred_labels[j][i])]))
            plt.imshow(inv_tensor.permute(1, 2, 0))
            if image_num == 0:
                if os.path.isdir(false_dir):
                    shutil.rmtree(false_dir)
            if not os.path.isdir(false_dir):
                os.mkdir(false_dir)
            fig.savefig(os.path.join(false_dir, image_name))
            image_num += 1


def create_training_list(args):
    training_list = []
    if args.train_photos:
        training_list.append('train_photos')
    if args.train_drawings:
        training_list.append('train_drawings')
    if args.train_images_with_drawings:
        training_list.append('train_images_with_drawings')
    return training_list


def checkpoint_dir_init(experiment_name, args):
    from datetime import datetime
    current_time = datetime.now().strftime("%H:%M:%S")
    if args.periods_classification and (args.train_periods_with_shapes_network):
        if args.sub_periods:
            sub = "sub_"
        else:
            sub = ""
        if args.train_periods_with_shapes_network:
            dir = "periods_classification/periods_with_shapes_" + args.shapes_net_type[:-11] + "/"
        else:
            dir = "periods_classification/periods/"
        task_type = dir + sub
        if args.part_train_set:
            task_type = task_type + "part_train_set_" + str(args.original_partial_train_set_ratio)
        if not os.path.isdir(task_type):
            os.mkdir(task_type)
        if not os.path.isdir(task_type + "/checkpoints/"):
            os.mkdir(task_type + "/checkpoints/")
        if not os.path.isdir(task_type + "/results/"):
            os.mkdir(task_type + "/results/")
        checkpoint_dir = os.path.join(os.getcwd(),
                                      task_type + "/checkpoints/" + experiment_name + "_" + str(current_time))
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            with open(checkpoint_dir + "/best_test_accuracy.txt", 'w') as f:
                f.write('0.0')
    else:
        if args.shapes_classification:
            task_type = "shapes_classification"
        if args.periods_classification:
            task_type = "periods_classification"
        if not os.path.isdir(task_type):
            os.mkdir(task_type)
        if not os.path.isdir(task_type + "/checkpoints/"):
            os.mkdir(task_type + "/checkpoints/")
        if not os.path.isdir(task_type + "/results/"):
            os.mkdir(task_type + "/results/")
        checkpoint_dir = os.path.join(os.getcwd(),
                                      task_type + "/checkpoints/" + experiment_name + "_" + str(current_time))

        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            with open(checkpoint_dir + "/best_test_accuracy.txt", 'w') as f:
                f.write('0.0')
    return checkpoint_dir


def reproducibility():
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    print_torch()
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def init_loss_parameters(training_procedure, args):
    if training_procedure == "train_images_with_drawings":
        if not args.classification_task:
            args.cross_entropy_loss_weight = 0.0
        if not args.force_embedding_similarity_task:
            args.similarity_loss_weight = 0.0
        if not args.image_to_drawings_generation_task:
            args.generation_loss_weight = 0.0
    return args


def find_classes(dir, args):
    import os
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx