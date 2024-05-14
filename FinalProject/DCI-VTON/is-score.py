import torch
import numpy as np
from scipy.stats import entropy
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

def inception_score(dataloader, model, splits=10):
    preds = []
    for images, _ in dataloader:
        with torch.no_grad():
            outputs = model(images)
            preds.append(softmax(outputs, dim=1).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    # Calculate score
    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
        p_y = np.mean(part, axis=0)
        scores = [entropy(p_yx, p_y) for p_yx in part]
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    # I took out the actual path to the generated images
    dataset = datasets.ImageFolder(root='path_to_generated_images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load pretrained Inception V3 model
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Sequential()  # Remove the fully connected layer
    inception_model = inception_model.to(device)

    # Calculate IS
    mean_is, std_is = inception_score(dataloader, inception_model)
    print(f"Inception Score: Mean = {mean_is}, Std = {std_is}")

if __name__ == "__main__":
    main()
