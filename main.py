from __future__ import print_function

import os.path

from utils_functions.imports import *
torch.multiprocessing.set_sharing_strategy('file_system')
from classify import *

def get_parser():
    parser = argparse.ArgumentParser('parameters', add_help=False)
    parser.add_argument('--eval_checkpoint', default='queries/images_with_drawings_shapes_checkpoints/', type=str, help="""""")
    parser.add_argument('--eval_classification', default=False, type=bool_flag, help="""""")
    parser.add_argument('--eval_retrieval', default=False, type=bool_flag, help="""""")
    parser.add_argument('--retrieval_k', default=10, type=int, help="""""")
    parser.add_argument('--eval_generation', default=False, type=bool_flag, help="""""")

    parser.add_argument('--RandomCrop', default=True, type=bool_flag, help="""""")
    parser.add_argument('--CropFactor', default=15, type=int, help="""""")
    parser.add_argument('--RandomHorizontalFlip', default=True, type=bool_flag, help="""""")
    parser.add_argument('--RandomVerticalFlip', default=False, type=bool_flag, help="""""")
    parser.add_argument('--RandomRotation', default=False, type=bool_flag, help="""""")
    parser.add_argument('--RandomRotationDegrees', default=5, type=int, help="""""")
    parser.add_argument('--GaussianBlur', default=False, type=bool_flag, help="""""")
    parser.add_argument('--GaussianBlurKernel', default=(5,9), type=int, help="""""")
    parser.add_argument('--GaussianBlurSigma', default=(0.1,3), type=int, help="""""")
    parser.add_argument('--RandomInvert', default=False, type=bool_flag, help="""""")

    parser.add_argument('--num_epochs', default=50, type=int, help="""""")
    parser.add_argument('--num_classes', default=10, type=int, help="""""")
    parser.add_argument('--batch_size', default=8, type=int, help="""""")
    parser.add_argument('--image_size', default=224, type=int, help="""""")
    parser.add_argument('--num_of_exps_to_eval', default=5, type=int, help="""""")
    parser.add_argument('--arch_type', default='pretrained_resnet101', type=str, help="""""") ##efficientnet, densenet ,resnet50, resnet152, glyphnet, pretrained_resnet50, coinnet, pretrained_coinnet
    parser.add_argument('--num_workers', default=8, type=int, help="""""")
    parser.add_argument('--photo_or_drawing_only_lr', default=0.00005, type=float, help="""""")
    parser.add_argument('--photo_or_drawing_only_weight_decay', default=0.00001, type=float, help="""""")
    parser.add_argument('--cssl_dataset_bast_dir', default="cssl_dataset", type=str, help="""""")

    parser.add_argument('--shapes_classification', default=True, type=bool_flag, help="""""")

    parser.add_argument('--periods_classification', default=False, type=bool_flag, help="""""")
    parser.add_argument('--train_periods_with_shapes_network', default=False, type=bool_flag, help="""""")
    parser.add_argument('--shapes_net_type', default="images_with_drawings_shapes_checkpoints", type=str, help="""""")
    parser.add_argument('--sub_periods', default=False, type=bool_flag, help="""""")

    parser.add_argument('--part_train_set', default=False, type=bool_flag, help="""""")
    parser.add_argument('--original_partial_train_set_ratio', default=4.0, type=float, help="""""")

    parser.add_argument('--train_photos', default=False, type=bool_flag, help="""""")
    parser.add_argument('--train_drawings', default=False, type=bool_flag, help="""""")
    parser.add_argument('--train_images_with_drawings', default=True, type=bool_flag, help="""""")

    parser.add_argument('--train_semi_supervised', default=True, type=bool_flag, help="""""")
    parser.add_argument('--semi_supervised_batch_size', default=8, type=int, help="""""")

    parser.add_argument('--classification_task', default=True, type=bool_flag, help="""""")
    parser.add_argument('--cross_entropy_loss_weight', default=0.05, type=float, help="""""")

    parser.add_argument('--force_embedding_similarity_task', default=True, type=bool_flag, help="""""")
    parser.add_argument('--similarity_loss_weight', default=0.85, type=float, help="""""")

    parser.add_argument('--image_to_drawings_generation_task', default=True, type=bool_flag, help="""""")
    parser.add_argument('--generation_print_results_every', default=51, type=int, help="""""")
    parser.add_argument('--generation_loss_weight', default=0.1, type=float, help="""""")
    parser.add_argument('--generation_l2_loss_weight', default=0.3, type=float, help="""""")
    parser.add_argument('--generation_perceptual_loss_weight', default=0.7, type=float, help="""""")
    return parser


def parse_args():
    parser = argparse.ArgumentParser('parameters', parents=[get_parser()])

    return parser.parse_args()


def run_shapes_with_periods(training_procedure, args, exp, experiment_name, run_wandb, checkpoint_dir):

    image_dirs_dict, draw_dirs_dict, image_checkpoint_dir = create_checkpoints_dirs_for_shapes_and_periods(
        args, exp, experiment_name)

    if training_procedure == "train_drawings":
        training_procedure = "train_drawings"
    elif training_procedure == "train_photos":
        training_procedure = "train_photos"
    if args.sub_periods:
        classes, class_to_idx_periods = find_classes(
            os.path.join(args.cssl_dataset_bast_dir, "sub_periods_with_shapes/drawings"), args)
    else:
        classes, class_to_idx_periods = find_classes(
            os.path.join(args.cssl_dataset_bast_dir, "periods_with_shapes/drawings"), args)

    train_dataloaders, test_dataloaders, dataset_sizes = data_loader_both_shapes_with_periods(args,
                                                                                                     args.image_size,
                                                                                                     draw_dirs_dict,
                                                                                                     image_dirs_dict,
                                                                                                     args.batch_size,
                                                                                                     data_transforms,
                                                                                                     args.num_workers,
                                                                                                     class_to_idx_periods)

    if args.eval_classification:
        eval_classification(args.arch_type, args.checkpoint_path, args, device, test_dataloaders, dataset_sizes, exp)
    print("train {} {}".format(training_procedure, exp))
    print("train_size: {}, test_size: {}".format(dataset_sizes['train'], dataset_sizes['test']))
    best_model_wts, best_experiment, model, best_acc = train_shapes_with_periods(
            args.arch_type, device,
            train_dataloaders, test_dataloaders, exp,
            dataset_sizes, checkpoint_dir, training_procedure,
            args,
            run_wandb, experiment_name)
    run_wandb.config.best_accuracy = best_acc
    return best_acc


def run_regular_shapes_or_periods(training_procedure, args, exp, experiment_name, run_wandb, checkpoint_dir):
    image_dir, draw_dir, draw_checkpoint_dir, image_checkpoint_dir = create_checkpoints_dirs(args, exp,
                                                                                             experiment_name)
    if args.shapes_classification:
        classes, class_to_idx = find_classes(os.path.join(args.cssl_dataset_bast_dir, "shape/drawings"), args)
    if args.periods_classification:
        if args.sub_periods:
            classes, class_to_idx = find_classes(
                os.path.join(args.cssl_dataset_bast_dir, "sub_periods_with_shapes/drawings"), args)
        else:
            classes, class_to_idx = find_classes(os.path.join(args.cssl_dataset_bast_dir, "periods_with_shapes/drawings"), args)

    data_dir = image_dir
    if training_procedure == "train_drawings":
        data_dir = draw_dir
        training_procedure = "train_drawings"
    elif training_procedure == "train_photos":
        data_dir = image_dir
        training_procedure = "train_photos"

    if training_procedure == "train_photos" or training_procedure == "train_drawings":
        dataloaders_train, dataloaders_test, dataset_sizes = data_loader(args,
                                                                               args.image_size, data_dir,
                                                                  args.batch_size,
                                                                  data_transforms,
                                                                  args.num_workers, class_to_idx)

    else:
        dataloaders_train, dataloaders_test, dataset_sizes = data_loader_both(args,
                                                                               args.image_size,
                                                                               image_dir, draw_dir,
                                                                               args.batch_size,
                                                                               data_transforms,
                                                                               args.num_workers, class_to_idx)
    to_train = True
    if args.eval_classification:
        eval_classification(args.arch_type, args.eval_checkpoint, args, device, dataloaders_test, dataset_sizes, exp)
        to_train = False
    if args.eval_retrieval:
        from retrieval import retrieval_calc_map
        retrieval_calc_map(args, args.eval_checkpoint, training_list, exp)
        to_train = False
    if args.eval_generation:
        to_train = False
        from generate import evaluation_of_generation
        evaluation_of_generation(device, exp, args.eval_checkpoint, args)
    print("train {} {}".format(training_procedure, exp))
    print("train_size: {}, test_size: {}".format(dataset_sizes['train'], dataset_sizes['test']))
    if to_train:
        best_model_wts, best_experiment, model_test, best_acc = train(args.arch_type, device, dataloaders_train, dataloaders_test, exp,
                                dataset_sizes, checkpoint_dir, draw_checkpoint_dir, training_procedure,
                                data_transforms, args, run_wandb)
        run_wandb.finish()
        del model_test
    return 0.0



if __name__ == '__main__':
    reproducibility()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    args, exp_list, project_name, data_transforms = init_experiments(args)
    training_list = create_training_list(args)
    if not os.path.isdir("periods_classification"):
        os.mkdir("periods_classification")
    if not os.path.isdir("shapes_classification"):
        os.mkdir("shapes_classification")
    for training_procedure in training_list:
        args = init_loss_parameters(training_procedure, args)
        for exp in exp_list:
            experiment_procedure, experiment_name = create_experiment_name(training_procedure, args, exp, False)
            run_wandb = wandb_init(args,
                                   training_procedure, exp, experiment_name, experiment_procedure, project_name)
            checkpoint_dir = checkpoint_dir_init(experiment_name, args)
            if args.train_periods_with_shapes_network and \
                    args.periods_classification:
                _ = run_shapes_with_periods(training_procedure, args, exp, experiment_name, run_wandb, checkpoint_dir)
            else:
                _ = run_regular_shapes_or_periods(training_procedure, args, exp, experiment_name, run_wandb, checkpoint_dir)