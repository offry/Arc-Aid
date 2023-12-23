from utils_functions.imports import *
from util_models.util_models import *
from utils_functions.util_functions import *
import torch.optim as optim

def load_drawings_network(exp, draw_checkpoint_dir, device, args):
    model_resnet = get_model(args.resnet_type, args.num_classes, args).to(device)
    filename = os.path.join(draw_checkpoint_dir + exp + "/" + exp + "_best_test_accuracy" + '.pth.tar')
    state_dict = torch.load(filename, map_location="cpu")
    model_resnet.load_state_dict(state_dict)
    return model_resnet, state_dict


def get_optimizer_criterion_scheduler(args, model, draw_or_photo):
    criterion = torch.nn.CrossEntropyLoss()
    dist_criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
    if draw_or_photo=="draw":
        optimizer = optim.Adam(model.parameters(), lr=args.draw_lr, betas=(0.9, 0.999), weight_decay=args.draw_weight_decay)
        # Decay LR by a factor of 0.1 every 3 epochs
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.draw_step_size, gamma=args.draw_gamma)
        schedulr = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
        # wandb.config.criterion, wandb.config.draw_optimizer, wandb.config.draw_scheduler = criterion, optimizer, scheduler
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.image_lr, betas=(0.9, 0.999), weight_decay=args.image_weight_decay)
        # Decay LR by a factor of 0.1 every 3 epochs
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.image_step_size, gamma=args.image_gamma)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
        # wandb.config.criterion, wandb.config.image_optimizer, wandb.config.image_scheduler = criterion, optimizer, scheduler
    return criterion, dist_criterion, optimizer, scheduler


def print_results(results, dataset_sizes, phase):
    epoch_results = {}
    for key in results:
        if key=="running_corrects":
            epoch_results['acc'] = 100 * results[key].float() / dataset_sizes[phase]
        elif key=="running_drawings_corrects":
            epoch_results['drawings_acc'] = 100 * results[key] / dataset_sizes[phase]
        elif key=='running_generation_corrects':
            epoch_results['running_generation_corrects'] = 100 * results[key] / dataset_sizes[phase]
        else:
            epoch_results[key] = results[key] / dataset_sizes[phase]
    for key in epoch_results:
        wandb.log({phase + " " + key: epoch_results[key]})

    if 'running_drawings_corrects' in results:
        print('{} Loss: {:.4f}, acc: {:.4f}, drawings_acc: {:.4f}'.format(phase, epoch_results['running_loss'], epoch_results['acc'],
                                                                          epoch_results['drawings_acc']))
    elif 'running_generation_corrects' in results:
        print('{} Loss: {:.4f}, acc: {:.4f}, generation_acc: {:.4f}'.format(phase, epoch_results['running_loss'],
                                                                          epoch_results['acc'],
                                                                          epoch_results['running_generation_corrects']))
    else:
        print('{} Loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_results['running_loss'], epoch_results['acc']))
    return epoch_results


def save_best_results(epoch, epoch_results, checkpoint_dir, exp, image_model, best_acc, best_model_wts, best_experiment, auto_model):
    if epoch == 0:
        best_acc = epoch_results['acc']
        # run_wandb.config.best_accuracy = best_acc
    if epoch_results['acc'] >= best_acc:
        best_acc = epoch_results['acc']
        # run_wandb.config.update({"best_accuracy": best_acc})
        # run_wandb.config.best_accuracy = best_acc
        print("new best image acc {}".format(best_acc))
        best_model_wts = copy.deepcopy(image_model.state_dict())
        with open(checkpoint_dir + "/best_test_accuracy.txt", 'r+') as f:
            lines = f.readlines()
        if float(lines[0]) <= best_acc:
            best_experiment = True
            with open(checkpoint_dir + "/best_test_accuracy.txt", 'r+') as f:
                f.write(str(best_acc.item()))
            print(
                "NEW BEST ACCURACY IN THIS KIND OF TRAINING PROCEDURE! Previous best: {} New best: {}".format(
                    float(lines[0]), best_acc))
            if auto_model is not None:
                filename_auto = os.path.join(checkpoint_dir, exp + "_best_auto_model" + '.pth.tar')
                torch.save(auto_model.state_dict(), filename_auto)
                filename = os.path.join(checkpoint_dir, exp + "_best_test_accuracy" + '.pth.tar')
                torch.save(auto_model.model.state_dict(), filename)
            else:
                filename = os.path.join(checkpoint_dir, exp + "_best_test_accuracy" + '.pth.tar')
                torch.save(image_model.state_dict(), filename)
    return best_model_wts, best_acc, best_experiment
