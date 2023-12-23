import kornia.filters
import torchvision.transforms.functional

from utils_functions.imports import *
from util_models.resnet_with_skip import *

def evaluation_of_generation(device, exp, checkpoint_path, args):
    exp_name = 'queries/' + str(exp)
    if not os.path.isdir("queries/results/"):
        os.mkdir("queries/results/")
    if not os.path.isdir("queries/results/generation"):
        os.mkdir("queries/results/generation")
    results_dir = "queries/results/generation/" + exp + "/"
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    resnet = models.resnet50(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 10)

    checkpoint = torch.load(checkpoint_path + args.arch_type + "_" + exp +
                            "/" + exp + "_best_auto_model.pth.tar", map_location="cpu")
    auto_images_model = Resnet_with_skip(resnet).to(device)
    auto_images_model.load_state_dict(checkpoint)
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    for image_path in os.listdir(exp_name):
        image = Image.open(exp_name + "/" + image_path)
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        image_tensor = transform(image)
        image_tensor = image_tensor.to(device)
        _, reconstruction = auto_images_model(image_tensor.unsqueeze(0))
        recon_tensor = reconstruction[0].repeat(3, 1, 1)
        plot_recon = recon_tensor.to("cpu").permute(1, 2, 0).detach().numpy()
        im_gray = cv2.cvtColor(plot_recon, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        im_bw = cv2.morphologyEx(im_gray, cv2.MORPH_CLOSE, kernel)
        (thresh, im_bw) = cv2.threshold(im_bw, 235, 255, cv2.THRESH_BINARY)
        black_pixels = np.where((im_bw[:, :] == 0))
        im_bw[black_pixels] = [45]
        plt.imshow(plot_recon)
        plt.savefig(
            results_dir + image_path + ".png")