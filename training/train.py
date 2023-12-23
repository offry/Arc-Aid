import kornia.losses
import pytorch_metric_learning.losses
import torch.nn

from utils_functions.imports import *
from util_models.util_models import *
from utils_functions.util_functions import *
from utils_functions.util_train_functions import *
from util_models.resnet_with_skip import Resnet_with_skip
from util_models.densenet_with_skip import Densenet_with_skip
from util_models.glyphnet_with_skip import Glyphnet_with_skip
from torchvision.models import vgg16_bn
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


def weights_init(m):
    if isinstance(m, (nn.Conv2d,)):
        if m.weight.shape==torch.Size([8,1,3,3]):
            return
        # torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        # torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

        # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        # torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        # torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def perceptual_loss_for_vgg(x,y, criterion, l, div, list_of_layers, perceptual_loss_dist_vgg):
    cur_embedding_vgg = nn.Sequential(nn.Sequential(*list_of_layers[:l+1]))
    cur_loss = -criterion(cur_embedding_vgg(x), cur_embedding_vgg(y)).mean()
    cur_loss = torch.div(torch.add(cur_loss, 1), div)
    cur_loss = torch.add(perceptual_loss_dist_vgg, cur_loss.cuda())
    return cur_loss


def train_drawings_or_photos_only(dataloaders_train, dataloaders_test, device, arch_type, model,
                                  args, dataset_sizes, checkpoint_dir, exp, run_wandb):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.photo_or_drawing_only_lr, betas=(0.9, 0.999),
                                 weight_decay=args.photo_or_drawing_only_weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    criterion = torch.nn.CrossEntropyLoss()
    best_experiment = False
    gray_scale = transforms.Grayscale(num_output_channels=1)
    for epoch in range(args.num_epochs):
        print('Epoch {}/{} for {}'.format(epoch, args.num_epochs - 1, arch_type))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            results = {}
            results['running_loss'], results['running_corrects'], i = 0.0, 0.0, 0
            if phase=='train':
                dataloaders = dataloaders_train
                model.train()
            else:
                model.eval()
                dataloaders = dataloaders_test
            for inputs, labels, _ in dataloaders:
                inputs = inputs.to(device)
                if isinstance(labels, list):
                    labels = np.asarray(labels)
                    labels = torch.from_numpy(labels)

                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                i += 1
                with torch.set_grad_enabled(phase == 'train'):
                    if "glyphnet" in arch_type:
                        outputs = model(gray_scale(inputs))
                    else:
                        outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    cross_entropy_loss = criterion(outputs, labels)
                    loss = cross_entropy_loss
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                results['running_loss'] += loss.item() * inputs.size(0)
                results['running_corrects'] += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
                wandb.log({"lr": optimizer.param_groups[0]["lr"]})

            epoch_results = print_results(results, dataset_sizes, phase)
            if phase == 'test':
                if epoch == 0:
                    best_acc = 0.0
                    best_model_wts = copy.deepcopy(model.state_dict())
                best_model_wts, best_acc, best_experiment = save_best_results(epoch, epoch_results, checkpoint_dir, exp,
                                                                              model, best_acc, best_model_wts,
                                                                              best_experiment, None)
    filename = os.path.join(checkpoint_dir, exp + "_best_test_accuracy" + '.pth.tar')
    checkpoint = torch.load(filename, map_location="cpu")
    model.load_state_dict(checkpoint)
    return best_model_wts, best_experiment, model, best_acc


def train_images_with_drawings(dataloaders_train, dataloaders_test, device, arch_type,
                               image_model, drawing_model, args, data_transforms,
                               dataset_sizes, checkpoint_dir, exp, run_wandb):
    dist_criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
    criterion = torch.nn.CrossEntropyLoss()
    best_experiment = False
    if args.shapes_classification:
        task_type = "shapes_classification"
    if args.periods_classification:
        task_type = "periods_classification"
    if args.train_semi_supervised:
        dataloader_all = data_loader_both_for_all_dist(args, args.image_size,
                                                       os.path.join(args.cssl_dataset_bast_dir, "all_image_base"),
                                                       os.path.join(args.cssl_dataset_bast_dir, "all_drawing_base"),
                                                                          args.semi_supervised_batch_size,
                                                                          data_transforms, dataloaders_test,
                                                                          dataloaders_train)
        dataloaders_t = dataloader_all
    else:
        dataloaders_t = dataloaders_train
    models_to_train = ["drawings", "photos"]
    if "resnet" in arch_type:
        auto_images_model = Resnet_with_skip(image_model.to("cpu")).to(device)
        # auto_images_model.decoder.apply(weights_init)
        auto_drawings_model = Resnet_with_skip(drawing_model.to("cpu")).to(device)
        # auto_drawings_model.decoder.apply(weights_init)
        embedding_image_model = torch.nn.Sequential(*(list(auto_images_model.model.children())[:-1]))
        embedding_drawing_model = torch.nn.Sequential(*(list(auto_drawings_model.model.children())[:-1]))
    elif "efficientnet" in arch_type:
        auto_images_model = Efficientnet_with_skip(image_model.to("cpu")).to(device)
        auto_images_model.decoder.apply(weights_init)
        auto_drawings_model = Efficientnet_with_skip(drawing_model.to("cpu")).to(device)
        auto_drawings_model.decoder.apply(weights_init)
        embedding_image_model = torch.nn.Sequential(*(list(auto_images_model.model.children())[:-2]))
        embedding_drawing_model = torch.nn.Sequential(*(list(auto_drawings_model.model.children())[:-2]))
    elif "densenet" in arch_type:
        auto_images_model = Densenet_with_skip(image_model.to("cpu")).to(device)
        auto_images_model.decoder.apply(weights_init)
        auto_drawings_model = Densenet_with_skip(drawing_model.to("cpu")).to(device)
        auto_drawings_model.decoder.apply(weights_init)
        embedding_image_model = torch.nn.Sequential(*(list(auto_images_model.model.children())[:-2]))
        embedding_drawing_model = torch.nn.Sequential(*(list(auto_drawings_model.model.children())[:-2]))
    elif "glyphnet" in arch_type:
        auto_images_model = Glyphnet_with_skip(image_model.to("cpu")).to(device)
        auto_images_model.decoder.apply(weights_init)
        auto_drawings_model = Glyphnet_with_skip(drawing_model.to("cpu")).to(device)
        auto_drawings_model.decoder.apply(weights_init)
        embedding_image_model = embedding_glyphnet(auto_images_model.model)
        embedding_drawing_model = embedding_glyphnet(auto_drawings_model.model)
    elif "coinnet" in arch_type:
        auto_images_model = Coinnet_with_skip(image_model.to("cpu")).to(device)
        auto_images_model.decoder.apply(weights_init)
        auto_drawings_model = Coinnet_with_skip(drawing_model.to("cpu")).to(device)
        auto_drawings_model.decoder.apply(weights_init)
        embedding_image_model = embedding_coinnet(image_model)
        embedding_drawing_model = embedding_coinnet(drawing_model)

    mse_criterion = nn.MSELoss()
    # mse_criterion = nn.L1Loss()
    vgg = vgg16_bn(pretrained=True).to(device)
    list_of_layers = list(vgg.features.children())
    list_of_layers_for_per = [22, 42]
    for param in vgg.parameters():
        param.requires_grad = False

    auto_images_optimizer = torch.optim.Adam(auto_images_model.parameters(), lr=args.photo_or_drawing_only_lr, betas=(0.9, 0.999),
                                             weight_decay=args.photo_or_drawing_only_weight_decay)
    auto_drawings_optimizer = torch.optim.Adam(auto_drawings_model.parameters(), lr=args.photo_or_drawing_only_lr, betas=(0.9, 0.999),
                                             weight_decay=args.photo_or_drawing_only_weight_decay)

    auto_images_scheduler = lr_scheduler.CosineAnnealingLR(auto_images_optimizer, T_max=args.num_epochs, eta_min=0)
    auto_drawings_scheduler = lr_scheduler.CosineAnnealingLR(auto_drawings_optimizer, T_max=args.num_epochs, eta_min=0)

    gray_scale = transforms.Grayscale(num_output_channels=1)
    gray_scale_three = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))

    for epoch in range(args.num_epochs):
        print('Epoch {}/{} for {}'.format(epoch, args.num_epochs - 1, arch_type))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            results = {}
            results['running_loss'], results['running_dist_loss'], results['running_cross_entropy_loss'], results[
                'running_corrects'], results['running_generation_loss'], results['running_generation_cross_entropy_loss'], i = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
            i = 0
            if phase == "train":
                dataloaders = dataloaders_t
            else:
                dataloaders = dataloaders_test
                auto_drawings_model.eval()
                auto_images_model.eval()
            for draw_data, image_data in dataloaders:
                draw_inputs, draw_labels, draw_paths = draw_data
                image_inputs, image_labels, image_paths = image_data
                draw_inputs = draw_inputs.to(device)
                image_inputs = image_inputs.to(device)
                draw_labels = draw_labels.to(device)
                image_labels = image_labels.to(device)

                # zero the parameter gradients
                auto_images_optimizer.zero_grad()
                auto_drawings_optimizer.zero_grad()
                for model_name in models_to_train:
                    if model_name == "drawings":
                        if phase != "train":
                            continue
                        with torch.set_grad_enabled(phase == 'train'):
                            auto_drawings_model.train()
                            auto_images_model.eval()
                            embedding_drawing_model.train()
                            embedding_image_model.eval()
                            if "glyphnet" in arch_type:
                                outputs, _, _, _, _, _, _ = auto_drawings_model.forward_pred(
                                    gray_scale(draw_inputs))
                            else:
                                outputs = auto_drawings_model.forward_pred(draw_inputs)
                            if args.train_semi_supervised:
                                mask_list = []
                                cross_entropy_loss = torch.zeros(1).to(device)
                                div = 0
                                for path, n in zip(draw_paths, range(len(draw_paths))):
                                    if path in dataloaders_train.dataset.draw_datasets.train_paths:
                                        index = dataloaders_train.dataset.draw_datasets.train_paths.index(
                                            path)
                                        label = torch.tensor(
                                            dataloaders_train.dataset.draw_datasets.train_label[index],
                                            dtype=torch.int64).to(device)
                                        cross_entropy_loss = torch.add(cross_entropy_loss,
                                                                           criterion(outputs[n], label))
                                        div += 1
                                        mask_list.append(True)
                                    else:
                                        mask_list.append(False)
                                if div != 0:
                                    cross_entropy_loss = torch.div(cross_entropy_loss, div)
                            else:
                                cross_entropy_loss = criterion(outputs, draw_labels)
                                div = 1
                            loss = cross_entropy_loss
                            if phase == 'train':
                                if div!=0:
                                    loss.backward()
                                    auto_drawings_optimizer.step()
                    elif model_name == "photos":
                        i += 1
                        auto_drawings_model.eval()
                        embedding_drawing_model.eval()
                        if phase=='train':
                            auto_images_model.train()
                            embedding_image_model.train()
                        else:
                            auto_images_model.eval()
                            embedding_image_model.eval()
                        with torch.no_grad():
                            if "glyphnet" in arch_type:
                                embedding_output_drawings = embedding_drawing_model(gray_scale(draw_inputs))
                            else:
                                embedding_output_drawings = embedding_drawing_model(draw_inputs)
                        with torch.set_grad_enabled(phase == 'train'):
                            if args.image_to_drawings_generation_task:
                                if "glyphnet" in arch_type:
                                    out_pred_model_output, x1, x2, x3, x4, x5, x6 = auto_images_model.forward_pred(
                                        gray_scale(image_inputs))
                                    output_auto_model_images = auto_images_model.forward_decode(
                                        x1, x2, x3, x4, x5, x6)
                                else:
                                    out_pred_model_output, output_auto_model_images = auto_images_model(image_inputs)
                                l2_loss = mse_criterion(kornia.enhance.invert(output_auto_model_images[:, 0, :, :]),
                                                        draw_inputs[:, 0, :, :])

                                # l2_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                #     kornia.enhance.invert(output_auto_model_images[:, 0, :, :]),
                                #     draw_inputs[:, 0, :, :])

                                generation_three = \
                                    gray_scale_three(kornia.enhance.invert(output_auto_model_images[:, 0, :, :]).unsqueeze(dim=1))

                                perceptual_loss_dist_vgg = torch.zeros(1).to(device)
                                for l in list_of_layers_for_per:
                                    perceptual_loss_dist_vgg = perceptual_loss_for_vgg(
                                        generation_three, draw_inputs, dist_criterion, l, len(list_of_layers_for_per),
                                        list_of_layers, perceptual_loss_dist_vgg)
                                perceptual_loss = perceptual_loss_dist_vgg.to(device)

                                generation_loss = args.generation_l2_loss_weight * l2_loss + \
                                                   args.generation_perceptual_loss_weight * perceptual_loss
                            else:
                                if "glyphnet" in arch_type:
                                    out_pred_model_output = auto_images_model.forward_pred(gray_scale(image_inputs))
                                else:
                                    out_pred_model_output = auto_images_model.forward_pred(image_inputs)
                                generation_loss = torch.zeros(1).to(device)

                            if args.force_embedding_similarity_task:
                                if "glyphnet" in arch_type:
                                    embedding_output_images = embedding_image_model(gray_scale(image_inputs))
                                else:
                                    embedding_output_images = embedding_image_model(image_inputs)
                                # similarity_loss = torch.zeros(1).to(device)
                                # for a in range(len(embedding_output_images)):
                                #     a_cur_loss = torch.zeros(1).to(device)
                                #     b_cur_loss = torch.zeros(1).to(device)
                                #     for b in range(len(embedding_output_images)):
                                #         if a == b:
                                #             a_cur_loss += torch.exp(
                                #                 torch.div(dist_criterion(embedding_output_images[a],
                                #                                           embedding_output_drawings[b]).mean(), 0.07))
                                #         if a!=b:
                                #             b_cur_loss += torch.exp(
                                #                 torch.div(dist_criterion(embedding_output_images[a],
                                #                                           embedding_output_drawings[b]).mean(), 0.07))
                                #
                                #     similarity_loss += -torch.log(torch.div(a_cur_loss, b_cur_loss))
                                # similarity_loss = torch.div(similarity_loss, len(embedding_output_images))
                                similarity_loss = -dist_criterion(embedding_output_images, embedding_output_drawings).mean()
                                similarity_loss = torch.add(similarity_loss, 1)
                            else:
                                similarity_loss = torch.zeros(1).to(device)
                            if phase=='train':
                                if args.train_semi_supervised:
                                    mask_list = []
                                    cross_entropy_loss = torch.zeros(1).to(device)
                                    div = 0
                                    preds = []
                                    for path, n in zip(image_paths, range(len(image_paths))):
                                        if path.split("/")[3] in dataloaders_train.dataset.image_datasets.train_paths:
                                            index = dataloaders_train.dataset.image_datasets.train_paths.index(path.split("/")[3])
                                            label = torch.tensor(dataloaders_train.dataset.image_datasets.train_label[index], dtype=torch.int64).to(device)
                                            cross_entropy_loss = torch.add(cross_entropy_loss,
                                                                               criterion(out_pred_model_output[n], label))
                                            div +=1
                                            mask_list.append(True)
                                        else:
                                            mask_list.append(False)
                                    if div!=0:
                                        cross_entropy_loss = torch.div(cross_entropy_loss, div)
                                else:
                                    cross_entropy_loss = criterion(out_pred_model_output, image_labels)
                                    div = 1
                            else:
                                _, preds = torch.max(out_pred_model_output, 1)
                            loss = generation_loss * args.generation_loss_weight + \
                                   cross_entropy_loss * args.cross_entropy_loss_weight + \
                                   args.similarity_loss_weight * similarity_loss

                            # backward + optimize only if in training phase

                            if phase == 'train':
                                loss.backward()
                                auto_images_optimizer.step()
                            if i%25==0:
                                print("batch {} out of {}, {}".format(i, len(dataloaders), phase))
                        # statistics

                        results['running_loss'] += loss.item() * image_inputs.size(0)
                        results['running_cross_entropy_loss'] += cross_entropy_loss.item() * image_inputs.size(0)
                        if phase=='test':
                            results['running_corrects'] += torch.sum(preds == image_labels.data)
                        results[
                            'running_dist_loss'] += similarity_loss.item() * image_inputs.size(
                            0)
                        results[
                            'running_generation_loss'] += generation_loss.item() * image_inputs.size(
                            0)
            if phase == 'train':
                auto_drawings_scheduler.step()
                auto_images_scheduler.step()
                wandb.log({"lr": auto_images_optimizer.param_groups[0]["lr"]})

            epoch_results = {}
            for key in results:
                if key == "running_corrects" and phase=='test':
                    epoch_results['acc'] = 100 * results[key].float() / dataset_sizes[phase]
                elif key == "running_corrects" and phase=='train':
                    epoch_results['acc'] = 0.0
                elif key == "running_drawings_corrects":
                    epoch_results['drawings_acc'] = 100 * results[key] / dataset_sizes[phase]
                else:
                    epoch_results[key] = results[key] / dataset_sizes[phase]
            for key in epoch_results:
                wandb.log({phase + " " + key: epoch_results[key]})

            if 'running_drawings_corrects' in results:
                print('{} Loss: {:.4f}, acc: {:.4f}, drawings_acc: {:.4f}'.format(phase, epoch_results['running_loss'],
                                                                                  epoch_results['acc'],
                                                                                  epoch_results['drawings_acc']))
            else:
                print('{} Loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_results['running_loss'], epoch_results['acc']))
            if phase == 'test':
                if epoch == 0:
                    best_acc = 0.0
                    best_model_wts = copy.deepcopy(auto_images_model.model.state_dict())
                best_model_wts, best_acc, best_experiment = save_best_results(epoch, epoch_results, checkpoint_dir, exp,
                                                                              auto_images_model.model, best_acc, best_model_wts,
                                                                              best_experiment, auto_images_model)
            filename_auto = os.path.join(checkpoint_dir, exp + "_last_auto_model" + '.pth.tar')
            torch.save(auto_images_model.state_dict(), filename_auto)
    filename_auto = os.path.join(checkpoint_dir, exp + "_best_auto_model" + '.pth.tar')
    checkpoint = torch.load(filename_auto, map_location="cpu")
    auto_images_model.load_state_dict(checkpoint)
    return best_model_wts, best_experiment, auto_images_model.model, best_acc


def train(arch_type, device, dataloaders_train, dataloaders_test, exp,
                                dataset_sizes, checkpoint_dir, draw_checkpoint_dir, training_procedure,
                                data_transforms, args, run_wandb):

    image_model = get_model(arch_type, args.num_classes, args).to(device)
    drawing_model = get_model(arch_type, args.num_classes, args).to(device)
    if training_procedure=="train_images_with_drawings":
        best_model_wts, best_experiment, auto_model, best_acc = train_images_with_drawings(dataloaders_train, dataloaders_test, device, arch_type,
                               image_model, drawing_model, args, data_transforms,
                               dataset_sizes, checkpoint_dir, exp, run_wandb)
    else:
        best_model_wts, best_experiment, auto_model, best_acc = train_drawings_or_photos_only(dataloaders_train, dataloaders_test, device, arch_type, image_model,
                                  args, dataset_sizes, checkpoint_dir, exp, run_wandb)
    return best_model_wts, best_experiment, auto_model, best_acc