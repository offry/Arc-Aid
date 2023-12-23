from utils_functions.util_functions import *
from util_models.util_models import *
from utils_functions.imports import *

def print_results(results, dataset_sizes, phase):
    epoch_results = {}
    if "train" in phase:
        dataset_phase = "train"
    else:
        dataset_phase = "test"
    for key in results:
        if key=="running_corrects":
            if isinstance(results[key], float):
                item = results[key]
            else:
                item = results[key].item()
            epoch_results['acc'] = 100 * item / dataset_sizes[dataset_phase]
        else:
            epoch_results[key] = results[key] / dataset_sizes[dataset_phase]
    for key in epoch_results:
        wandb.log({phase + " " + key: epoch_results[key]})

    print('{} Loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_results['running_loss'], epoch_results['acc']))
    return epoch_results


def save_best_results(epoch, epoch_results, checkpoint_dir, exp, image_model, best_acc, best_model_wts, best_experiment,
                      auto_model, run_wandb):
    if epoch == 0:
        best_acc = epoch_results['acc']
    if epoch_results['acc'] >= best_acc:
        best_acc = epoch_results['acc']
        print("new best image acc {}".format(best_acc))
        best_model_wts = copy.deepcopy(image_model.state_dict())
        with open(checkpoint_dir + "/best_test_accuracy.txt", 'r+') as f:
            lines = f.readlines()
        if float(lines[0]) <= best_acc:
            best_experiment = True
            with open(checkpoint_dir + "/best_test_accuracy.txt", 'r+') as f:
                f.write(str(best_acc))
            print(
                "NEW BEST ACCURACY IN THIS KIND OF TRAINING PROCEDURE! Previous best: {} New best: {}".format(
                    float(lines[0]), best_acc))
            if auto_model is not None:
                filename_auto = os.path.join(checkpoint_dir, exp + '_best_auto_model.pth.tar')
                torch.save(auto_model.state_dict(), filename_auto)
            else:
                filename = os.path.join(checkpoint_dir, exp + '_best_test_accuracy.pth.tar')
                torch.save(image_model.state_dict(), filename)
    return best_model_wts, best_acc, best_experiment


def train_images_with_drawings_shapes_with_periods(train_dataloaders, test_dataloaders, device, arch_type, periods_model,
                                  args, dataset_sizes, checkpoint_dir, exp, run_wandb):
    optimizer = torch.optim.Adam(periods_model.parameters(),
                                                          lr=args.photo_or_drawing_only_lr,
                                                          betas=(0.9, 0.999),
                                                          weight_decay=args.photo_or_drawing_only_weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=args.num_epochs,
                                                            eta_min=0)
    criterion = torch.nn.CrossEntropyLoss()
    best_experiment, best_model_wts, best_acc = False, None, 0.0
    gray_scale = transforms.Grayscale(num_output_channels=1)
    for epoch in range(args.num_epochs):
        print('Epoch {}/{} for {}'.format(epoch, args.num_epochs - 1, arch_type))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            results = {}
            results['running_loss'], results['running_corrects'], i = 0.0, 0.0, 0
            if phase == 'train':
                dataloaders = train_dataloaders
                periods_model.train()
            else:
                periods_model.eval()
                dataloaders = test_dataloaders
            for draw_data, image_data in dataloaders:
                draw_inputs, draw_labels, _ = draw_data
                image_inputs, image_labels, path = image_data
                if args.shapes_net_type=="drawings_shapes_checkpoints":
                    inputs = draw_inputs.to(device)
                    labels = draw_labels.to(device)
                else:
                    inputs = image_inputs.to(device)
                    labels = image_labels.to(device)
                # zero the parameter gradients
                i += 1
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if 'glyphnet' in arch_type:
                        outputs = periods_model(gray_scale(inputs))
                    else:
                        outputs = periods_model(inputs)
                    _, preds = torch.max(outputs, 1)
                    cross_entropy_loss = criterion(outputs, labels)
                    loss = cross_entropy_loss
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                results['running_loss'] += loss.item() * inputs.size(0)
                results['running_corrects'] += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
                wandb.log({"backbone_lr": optimizer.param_groups[0]["lr"]})

            epoch_results = print_results(results, dataset_sizes, phase)
            if phase == 'test':
                if epoch == 0:
                    best_acc = 0.0
                    best_model_wts = copy.deepcopy(periods_model.state_dict())
                best_model_wts, best_acc, _ = save_best_results(epoch,
                                                                      epoch_results,
                                                                      checkpoint_dir, exp,
                                                                      periods_model,
                                                                      best_acc,
                                                                      best_model_wts,
                                                                      best_experiment, None, run_wandb)
    return best_model_wts, best_experiment, periods_model, best_acc


def train_shapes_with_periods(arch_type, device, train_dataloaders, test_dataloaders, exp,
                                dataset_sizes, checkpoint_dir, training_procedure, args,
                                run_wandb, experiment_name):

    exp_number ="experiment_" + experiment_name.split("_")[2]
    if "resnet" in arch_type:
        if "pretrained" in arch_type:
            exp_number = "experiment_" + experiment_name.split("_")[3]
            arch_dir = "pretrained_resnet50_"
        else:
            arch_dir = "resnet50_"
    elif "glyphnet" in arch_type:
        arch_dir = "glyphnet_"
    elif "coinnet" in arch_type:
        arch_dir = "coinnet_"
    elif "densenet" in arch_type:
        arch_dir = "densenet_"
    if args.part_train_set:
        checkpoint_path = "periods_classification/" + args.shapes_net_type + "/" + arch_dir + exp_number + \
                          "part_train_set" + str(args.original_partial_train_set_ratio) + "/" + exp_number + "_best_test_accuracy.pth.tar"
    else:
        checkpoint_path = "periods_classification/" + args.shapes_net_type + "/" + arch_dir + exp_number + "/" \
                          + exp_number + "_best_test_accuracy.pth.tar"

    periods_model = get_model(arch_type, 10, args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    periods_model.to(device)

    periods_model.load_state_dict(checkpoint)

    for param in periods_model.parameters():
        param.requires_grad = False
    if "resnet" in arch_type:
        for param in periods_model.layer4.parameters():
            param.requires_grad = True
        if args.sub_periods:
            periods_model.fc = nn.Linear(2048, 5).to(device)
        else:
            periods_model.fc = nn.Linear(2048, 3).to(device)
    elif "densenet" in arch_type:
        for param in periods_model.features.denseblock4.parameters():
            param.requires_grad = True
        if args.sub_periods:
            periods_model.classifier = nn.Linear(2208, 5).to(device)
        else:
            periods_model.classifier = nn.Linear(2208, 3).to(device)
    elif "glyphnet" in arch_type:
        for param in periods_model.final_block.parameters():
            param.requires_grad = True
        for param in periods_model.inner_blocks.parameters():
            param.requires_grad = True
        if args.sub_periods:
            periods_model.final_block.fully_connected = nn.Linear(512, 5).to(device)
        else:
            periods_model.final_block.fully_connected = nn.Linear(512, 3).to(device)
    elif "coinnet" in arch_type:
        for param in periods_model.module.conv_block.parameters():
            param.requires_grad = True
        for param in periods_model.module.cbp_layer_feat.parameters():
            param.requires_grad = True
        for param in periods_model.module.model_d161.features.denseblock4.parameters():
            param.requires_grad = True
        if args.sub_periods:
            periods_model.module.fc = nn.Linear(2208, 5).to(device)
        else:
            periods_model.module.fc = nn.Linear(2208, 3).to(device)

    best_model_wts, best_experiment, model, best_acc = \
        train_images_with_drawings_shapes_with_periods(train_dataloaders,
                              test_dataloaders, device, arch_type, periods_model,
                              args, dataset_sizes, checkpoint_dir, exp, run_wandb)


    return best_model_wts, best_experiment, model, best_acc