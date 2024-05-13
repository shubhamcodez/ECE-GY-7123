import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2), disp=False)[0]

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def get_activations(dataloader, model, dims=2048):
    model.eval()
    activations = []
    for images, _ in dataloader:
        with torch.no_grad():
            pred = model(images)[0]
        if pred.size(1) != dims:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(dims,))
        activations.append(pred.cpu().numpy())
    return np.concatenate(activations, axis=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    # I took out the actual paths to the real and generated images
    real_set = datasets.ImageFolder(root='path_to_real_images', transform=transform)
    gen_set = datasets.ImageFolder(root='path_to_generated_images', transform=transform)

    real_loader = DataLoader(real_set, batch_size=batch_size, shuffle=False)
    gen_loader = DataLoader(gen_set, batch_size=batch_size, shuffle=False)

    # Load pretrained Inception V3 model
    inception_model = models.inception_v3(pretrained=True)
    inception_model.fc = torch.nn.Sequential()  # Remove the fully connected layer
    inception_model = inception_model.to(device)

    # Get activations
    real_activations = get_activations(real_loader, inception_model)
    gen_activations = get_activations(gen_loader, inception_model)

    # Calculate FID
    fid_value = calculate_fid(real_activations, gen_activations)
    print(f"FID: {fid_value}")

if __name__ == "__main__":
    main()
