import random
import imageio
import torchvision.transforms.functional

from utils_functions.imports import *
import torch.nn as nn

class loader():
    def __init__(self, root, experiment_dir, data_transforms, class_to_idx, is_train=True, data_len=None, args = None, size = None):
        self.root = root
        self.is_train = is_train
        self.data_transforms = data_transforms
        self.experiment_dir = experiment_dir
        self.args = args
        self.size = size

        train_val_file = open(os.path.join(self.experiment_dir, 'split.txt'))
        img_name_list, label_list, train_test_list = [], [], []
        for line in train_val_file:
            if line=='\n':
                continue
            line = line.split('*')
            img_name_list.append(line[0])
            label_list.append(line[1])
            train_test_list.append(line[2][:-1])
        train_val_file.close()
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i=='train']
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if i=='test']
        if self.is_train:
            self.train_img = [os.path.join(self.root, '1', train_file) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [class_to_idx[x] for i, x in zip(train_test_list, label_list) if i=='train'][:data_len]
            self.train_paths = [os.path.join(self.root, '1', train_file) for train_file in
                              train_file_list[:data_len]]
        if not self.is_train:
            self.test_img = [os.path.join(self.root, '1', test_file) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [class_to_idx[x] for i, x in zip(train_test_list, label_list) if i=='test'][:data_len]
            self.test_paths = [os.path.join(self.root, '1', test_file) for test_file in
                                test_file_list[:data_len]]
    def transform(self, image_tuple, is_train=False):
        if self.args.RandomCrop and is_train:
            resize = transforms.Resize(size=(self.size+self.args.CropFactor, self.size+self.args.CropFactor))
        else:
            resize = transforms.Resize(size=(self.size, self.size))
        if image_tuple[0].mode!="L" and image_tuple[0].mode!="RGB" and image_tuple[0].mode!="RGBA":
            img = image_tuple[0]
            img_new = img.convert("L")
            image = resize(img_new)
        else:
            image = resize(image_tuple[0])

        grayscale = transforms.Grayscale(num_output_channels=3)
        # grayscale = transforms.Grayscale(num_output_channels=1)
        image = grayscale(image)
        if self.args.RandomCrop and is_train:
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.size, self.size))
            image = TF.crop(image, i, j, h, w)
        if self.args.GaussianBlur and is_train:
            if random.random() >  0.5:
                gaussian_blur = transforms.GaussianBlur(kernel_size=self.args.GaussianBlurKernel,
                                                        sigma=self.args.GaussianBlurSigma)
                image = gaussian_blur(image)
        if self.args.RandomHorizontalFlip and is_train:
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
        if self.args.RandomVerticalFlip and is_train:
            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
        if self.args.RandomRotation and is_train:
            rotation = transforms.RandomRotation(degrees=(0, self.args.RandomRotationDegrees))
            image = rotation(image)
        if self.args.RandomInvert and is_train:
            if random.random() > 0.5:
                image = TF.invert(image)
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # normalize = transforms.Normalize([0.5], [0.5])
        # Transform to tensor
        image = TF.to_tensor(image)

        image = normalize(image)
        image_tuple = (image, image_tuple[1], image_tuple[2])
        return image_tuple


    def __getitem__(self, index):
        if self.is_train:
            img, target, path = Image.open(self.train_img[index]), self.train_label[index], self.train_paths[index]
        else:
            img, target, path = Image.open(self.test_img[index]), self.test_label[index], self.test_paths[index]

        if self.args!=None:
            if self.is_train:
                x = self.transform((img, target, path), is_train=True)
                return x
            else:
                x = self.transform((img, target, path), is_train=True)
                return x
        return (img, target, path)

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class PathImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple_of_image_folder = super(PathImageFolder, self).__getitem__(index)
        path_to_return = self.imgs[index][0]
        new_tuple = (original_tuple_of_image_folder + (path_to_return,))
        return new_tuple


class DrawImagePairDataset(Dataset):
    def  __init__(self, args, size, draw_datasets, image_datasets, test=False):
        self.draw_datasets = draw_datasets  # datasets should be sorted!
        self.image_datasets = image_datasets
        self.test = test
        self.args = args
        self.size = size

    def transform(self, image_tuple, draw_tuple, is_train=False):
        if self.args.RandomCrop and is_train:
            resize = transforms.Resize(size=(self.size+self.args.CropFactor, self.size+self.args.CropFactor))
        else:
            resize = transforms.Resize(size=(self.size, self.size))
        if image_tuple[0].mode!="L" and image_tuple[0].mode!="RGB" and image_tuple[0].mode!="RGBA":
            img = image_tuple[0]
            img_new = img.convert("L")
            image = resize(img_new)
        else:
            image = resize(image_tuple[0])

        if draw_tuple[0].mode!="L" and draw_tuple[0].mode!="RGB" and draw_tuple[0].mode!="RGBA":
            draw = draw_tuple[0]
            draw_new = draw.convert("L")
            draw = resize(draw_new)
        else:
            draw = resize(draw_tuple[0])


        grayscale = transforms.Grayscale(num_output_channels=3)
        # grayscale = transforms.Grayscale(num_output_channels=1)
        image, draw = grayscale(image), grayscale(draw)
        if self.args.RandomCrop and is_train:
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.size, self.size))
            image = TF.crop(image, i, j, h, w)
            draw = TF.crop(draw, i, j, h, w)
        if self.args.GaussianBlur and is_train:
            if random.random() >  0.5:
                gaussian_blur = transforms.GaussianBlur(kernel_size=self.args.GaussianBlurKernel,
                                                        sigma=self.args.GaussianBlurSigma)
                # image, draw = gaussian_blur(image), gaussian_blur(draw)
                image, draw = gaussian_blur(image), draw
        if self.args.RandomHorizontalFlip and is_train:
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                draw = TF.hflip(draw)
        if self.args.RandomVerticalFlip and is_train:
            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                draw = TF.vflip(draw)
        if self.args.RandomRotation and is_train:
            rotation = transforms.RandomRotation(degrees=(0, self.args.RandomRotationDegrees))
            image, draw = rotation(image), rotation(draw)
        if self.args.RandomInvert and is_train:
            if random.random() > 0.5:
                # image, draw = TF.invert(image), TF.invert(draw)
                image, draw = TF.invert(image), draw
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # normalize = transforms.Normalize([0.5], [0.5])
        # Transform to tensor
        image = TF.to_tensor(image)
        draw = TF.to_tensor(draw)

        image, draw = normalize(image), normalize(draw)
        image_tuple, draw_tuple = (image, image_tuple[1], image_tuple[2]), (draw, draw_tuple[1], draw_tuple[2])
        return image_tuple, draw_tuple

    def __getitem__(self, index):
        x1 = self.draw_datasets[index]
        x2 = self.image_datasets[index]
        if self.test:
            return self.transform(x1, x2)
        else:
            x1, x2 = self.transform(x1, x2, is_train=True)
            return x1, x2

    def __len__(self):
        return len(self.draw_datasets)  # assuming both datasets have same length


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_transforms_arch(size, args, pil=False):
    if pil:
        p_pil = 1
    else:
        p_pil = 0
    if args.RandomCrop:
        crop_factor = args.CropFactor
    else:
        crop_factor = 0
    if args.RandomHorizontalFlip:
        p_RandomHorizontalFlip = 0.5
    else:
        p_RandomHorizontalFlip = 0
    if args.RandomVerticalFlip:
        p_RandomVerticalFlip = 0.5
    else:
        p_RandomVerticalFlip = 0
    if args.RandomInvert:
        p_RandomInvert = 0.5
    else:
        p_RandomInvert = 0
    if args.GaussianBlur:
        p_GaussianBlur = 0.7
    else:
        p_GaussianBlur = 0
    if args.RandomRotation:
        p_RandomRotation= 1
    else:
        p_RandomRotation = 0
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((size+crop_factor, size+crop_factor)),
            transforms.Grayscale(num_output_channels=3),
            # transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop((size,size)),
            # transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=p_RandomHorizontalFlip),
            transforms.RandomVerticalFlip(p=p_RandomVerticalFlip),
            transforms.RandomInvert(p=p_RandomInvert),
            transforms.RandomApply([transforms.RandomRotation(degrees=(0,args.RandomRotationDegrees))],
                                   p=p_RandomRotation),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=args.GaussianBlurKernel,
                                                    sigma=args.GaussianBlurSigma)], p=p_GaussianBlur),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # transforms.Normalize([0.5], [0.5])
        ]),
        'test': transforms.Compose([
            transforms.RandomApply([transforms.ToPILImage(),], p=p_pil),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # transforms.Normalize([0.5], [0.5])
        ]),
    }
    return data_transforms


def data_loader(args, size, experiment_dir, batch_size, data_transforms, num_workers, class_to_idx, recon=False):
    image_datasets = {}
    if "drawing" in experiment_dir:
        data_dir = os.path.join(args.cssl_dataset_bast_dir, "all_drawing_base")
    else:
        data_dir = os.path.join(args.cssl_dataset_bast_dir, "all_image_base")
    if recon:
        data_dir = "queries/all_dexi"
    image_datasets['train'] = loader(data_dir, experiment_dir, data_transforms['train'], class_to_idx, is_train=True, data_len=None,
                                     args=args, size=size)
    image_datasets['test'] = loader(data_dir, experiment_dir,
                                         data_transforms['test'], class_to_idx, is_train=False, data_len=None, args=args, size=size)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers, pin_memory=False)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    return dataloaders['train'], dataloaders['test'], dataset_sizes


def data_loader_both(args, size, image_experiment_dir, draw_experiment_dir, batch_size, data_transforms,
                     num_workers, class_to_idx, recon=False):
    import os
    draw_dir, image_dir = os.path.join(args.cssl_dataset_bast_dir, "all_drawing_base"), \
                          os.path.join(args.cssl_dataset_bast_dir, "all_image_base")
    # train_image_dataset = PathImageFolder(os.path.join(image_dir, 'train'))
    # train_draw_dataset = PathImageFolder(os.path.join(draw_dir, 'train'))
    train_image_dataset = loader(image_dir, image_experiment_dir, data_transforms['train'], class_to_idx, is_train=True, data_len=None)
    train_draw_dataset = loader(draw_dir, draw_experiment_dir,
                                    data_transforms['train'], class_to_idx, is_train=True, data_len=None)
    # test_image_dataset = PathImageFolder(os.path.join(image_dir, 'test'), data_transforms['test'])
    # test_draw_dataset = PathImageFolder(os.path.join(draw_dir, 'test'), data_transforms['test'])
    test_image_dataset = loader(image_dir, image_experiment_dir, data_transforms['test'], class_to_idx, is_train=False, data_len=None)
    test_draw_dataset = loader(draw_dir, draw_experiment_dir,
                                    data_transforms['test'], class_to_idx, is_train=False, data_len=None)
    dataset_train = DrawImagePairDataset(args, size, train_draw_dataset, train_image_dataset, test=False)
    dataset_test = DrawImagePairDataset(args, size, test_draw_dataset, test_image_dataset, test=True)

    g = torch.Generator()
    g.manual_seed(0)

    dataloaders_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False,
                                                    worker_init_fn=seed_worker, generator=g,)
    dataloaders_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=False,
                                                    worker_init_fn=seed_worker, generator=g,)
    dataset_sizes = {}
    dataset_sizes['train'], dataset_sizes['test'] = len(train_image_dataset), len(test_image_dataset)
    return dataloaders_train, dataloaders_test, dataset_sizes

class MyDataset(Dataset):
    def __init__(self, draw_datasets, image_datasets):
        self.draw_datasets = draw_datasets  # datasets should be sorted!
        self.image_datasets = image_datasets

    def __getitem__(self, index):
        x1 = self.draw_datasets[index]
        x2 = self.image_datasets[index]
        return x1, x2

    def __len__(self):
        return len(self.draw_datasets)  # assuming both datasets have same length

def data_loader_both_for_all_dist(args, size, image_dir, draw_dir, batch_size, data_transforms, dataloader_test,
                                  dataloader_train_labeled):
    print("IN THIS EXPERIMENT THE SEMI SUPERVISED IS ONE STAGE")
    # image_datasets = datasets.ImageFolder(image_dir, data_transforms['train'])
    #
    # dataset_sizes = len(image_datasets)

    train_image_dataset = PathImageFolder(image_dir)
    train_draw_dataset = PathImageFolder(draw_dir)
    # draw_dir, image_dir = "cssl_dataset/all_drawing_base/1", "cssl_dataset/all_image_base/1"
    # train_image_dataset = loader(image_dir, image_experiment_dir, data_transforms['train'], is_train=True, data_len=None)
    # train_draw_dataset = loader(draw_dir, draw_experiment_dir,
    #                                 data_transforms['train'], is_train=True, data_len=None)
    dataset_train = DrawImagePairDataset(args, size, train_draw_dataset, train_image_dataset, test=False)

    dataset_train_final = copy.deepcopy(dataset_train)
    image_test_list = dataloader_test.dataset.image_datasets.test_paths
    image_train_labeled_list = dataloader_train_labeled.dataset.image_datasets.train_paths
    for i in range(len(image_test_list)):
        image_test_list[i] = image_test_list[i].split("/")[3]
    for i in range(len(image_train_labeled_list)):
        image_train_labeled_list[i] = image_train_labeled_list[i].split("/")[3]
    for i in dataset_train.image_datasets.targets:
        if i !=0:
            a = 0
    for image, draw in zip(dataset_train.image_datasets.imgs, dataset_train.draw_datasets.imgs):
        image_name = image[0].split("/")
        for im in image_name:
            if "Base" in im:
                image_name = im
                break
        if image_name in image_test_list:
            dataset_train_final.image_datasets.imgs.remove((image[0], image[1]))
            dataset_train_final.draw_datasets.imgs.remove((draw[0], draw[1]))

        elif image_name in image_train_labeled_list:
            # dataset_train_final.image_datasets.imgs.remove((image[0], image[1]))
            # dataset_train_final.draw_datasets.imgs.remove((draw[0], draw[1]))
            continue

    g = torch.Generator()
    g.manual_seed(0)

    dataloaders_train = torch.utils.data.DataLoader(dataset_train_final, batch_size=batch_size, shuffle=True, num_workers=16,
                                                    worker_init_fn=seed_worker,
                                                    generator=g,)

    return dataloaders_train

class loader_multi_class():
    def __init__(self, root, experiment_dir, data_transforms, class_to_idx, is_train=True, data_len=None, args = None, size = None):
        self.root = root
        self.is_train = is_train
        self.data_transforms = data_transforms
        self.experiment_dir = experiment_dir
        self.args = args
        self.size = size

        train_val_file = open(os.path.join(self.experiment_dir, 'split.txt'))
        img_name_list, label_list, train_test_list = [], [], []
        for line in train_val_file:
            if line=='\n':
                continue
            line = line.split('*')
            img_name_list.append(line[0])
            label_list.append(line[1:len(line)-1])
            train_test_list.append(line[-1][:-1])
        train_val_file.close()
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i=='train']
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if i=='test']
        if self.is_train:
            self.train_img = [os.path.join(self.root, '1', train_file) for train_file in
                              train_file_list[:data_len]]
            self.train_label = []
            for i, x in zip(train_test_list, label_list):
                if i=='train':
                    cur_labels = [0]*len(class_to_idx)
                    for label in x:
                        if "_" in label and label not in class_to_idx.keys():
                            cur_labels[class_to_idx[label.split("_")[0]]] = 1
                        else:
                            cur_labels[class_to_idx[label]] = 1
                    self.train_label.append(cur_labels)
            # self.train_label = [class_to_idx[x] for i, x in zip(train_test_list, label_list) if i=='train'][:data_len]
            self.train_paths = [os.path.join(self.root, '1', train_file) for train_file in
                              train_file_list[:data_len]]
        if not self.is_train:
            self.test_img = [os.path.join(self.root, '1', test_file) for test_file in
                             test_file_list[:data_len]]
            self.test_label = []
            for i, x in zip(train_test_list, label_list):
                if i=='test':
                    cur_labels = [0]*len(class_to_idx)
                    for label in x:
                        if "_" in label and label not in class_to_idx.keys():
                            cur_labels[class_to_idx[label.split("_")[0]]] = 1
                        else:
                            cur_labels[class_to_idx[label]] = 1
                    self.test_label.append(cur_labels)
            # self.test_label = [class_to_idx[x] for i, x in zip(train_test_list, label_list) if i=='test'][:data_len]
            self.test_paths = [os.path.join(self.root, '1', test_file) for test_file in
                                test_file_list[:data_len]]


    def transform(self, image_tuple, is_train=False):
        if self.args.RandomCrop and is_train:
            resize = transforms.Resize(size=(self.size+self.args.CropFactor, self.size+self.args.CropFactor))
        else:
            resize = transforms.Resize(size=(self.size, self.size))
        if image_tuple[0].mode!="L" and image_tuple[0].mode!="RGB" and image_tuple[0].mode!="RGBA":
            img = image_tuple[0]
            img_new = img.convert("L")
            image = resize(img_new)
        else:
            image = resize(image_tuple[0])

        grayscale = transforms.Grayscale(num_output_channels=3)
        # grayscale = transforms.Grayscale(num_output_channels=1)
        image = grayscale(image)
        if self.args.RandomCrop and is_train:
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.size, self.size))
            image = TF.crop(image, i, j, h, w)
        if self.args.GaussianBlur and is_train:
            if random.random() >  0.5:
                gaussian_blur = transforms.GaussianBlur(kernel_size=self.args.GaussianBlurKernel,
                                                        sigma=self.args.GaussianBlurSigma)
                image = gaussian_blur(image)
        if self.args.RandomHorizontalFlip and is_train:
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
        if self.args.RandomVerticalFlip and is_train:
            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
        if self.args.RandomRotation and is_train:
            rotation = transforms.RandomRotation(degrees=(0, self.args.RandomRotationDegrees))
            image = rotation(image)
        if self.args.RandomInvert and is_train:
            if random.random() > 0.8:
                image = TF.invert(image)
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # normalize = transforms.Normalize([0.5], [0.5])
        # Transform to tensor
        image = TF.to_tensor(image)

        image = normalize(image)
        image_tuple = (image, image_tuple[1], image_tuple[2])
        return image_tuple


    def __getitem__(self, index):
        if self.is_train:
            img, target, path = Image.open(self.train_img[index]), self.train_label[index], self.train_paths[index]
        else:
            img, target, path = Image.open(self.test_img[index]), self.test_label[index], self.test_paths[index]

        if self.args!=None:
            if self.is_train:
                x = self.transform((img, target, path), is_train=True)
                return x
            else:
                x = self.transform((img, target, path), is_train=True)
                return x
        return (img, target, path)

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

def data_loader_multi_class(args, size, experiment_dir, batch_size, data_transforms, num_workers, class_to_idx):
    image_datasets = {}
    if "drawing" in experiment_dir:
        data_dir = os.path.join(args.cssl_dataset_bast_dir, "all_drawing_base")
    else:
        data_dir = os.path.join(args.cssl_dataset_bast_dir, "all_image_base")
    image_datasets['train'] = loader_multi_class(data_dir, experiment_dir, data_transforms['train'], class_to_idx, is_train=True, data_len=None,
                                     args=args, size=size)
    image_datasets['test'] = loader_multi_class(data_dir, experiment_dir,
                                         data_transforms['test'], class_to_idx, is_train=False, data_len=None, args=args, size=size)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers, pin_memory=False)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    return dataloaders['train'], dataloaders['test'], dataset_sizes


def data_loader_both_multi_class(args, size, image_experiment_dir, draw_experiment_dir, batch_size, data_transforms, num_workers, class_to_idx):
    draw_dir, image_dir = os.path.join(args.cssl_dataset_bast_dir, "all_drawing_base"), \
                          os.path.join(args.cssl_dataset_bast_dir, "all_image_base")
    # train_image_dataset = PathImageFolder(os.path.join(image_dir, 'train'))
    # train_draw_dataset = PathImageFolder(os.path.join(draw_dir, 'train'))
    train_image_dataset = loader_multi_class(image_dir, image_experiment_dir, data_transforms['train'], class_to_idx, is_train=True, data_len=None)
    train_draw_dataset = loader_multi_class(draw_dir, draw_experiment_dir,
                                    data_transforms['train'], class_to_idx, is_train=True, data_len=None)
    # test_image_dataset = PathImageFolder(os.path.join(image_dir, 'test'), data_transforms['test'])
    # test_draw_dataset = PathImageFolder(os.path.join(draw_dir, 'test'), data_transforms['test'])
    test_image_dataset = loader_multi_class(image_dir, image_experiment_dir, data_transforms['test'], class_to_idx, is_train=False, data_len=None)
    test_draw_dataset = loader_multi_class(draw_dir, draw_experiment_dir,
                                    data_transforms['test'], class_to_idx, is_train=False, data_len=None)
    dataset_train = DrawImagePairDataset(args, size, train_draw_dataset, train_image_dataset, test=False)
    dataset_test = DrawImagePairDataset(args, size, test_draw_dataset, test_image_dataset, test=True)

    g = torch.Generator()
    g.manual_seed(0)

    dataloaders_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False,
                                                    worker_init_fn=seed_worker, generator=g,)
    dataloaders_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False,
                                                    worker_init_fn=seed_worker, generator=g,)
    dataset_sizes = {}
    dataset_sizes['train'], dataset_sizes['test'] = len(train_image_dataset), len(test_image_dataset)
    return dataloaders_train, dataloaders_test, dataset_sizes