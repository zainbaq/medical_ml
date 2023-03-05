import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from xavier.modeling import AlzheimersMRIClassification, fit
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_image_directory', type=str, default='./data/mri_images/train/', required=True)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--output_path', type=str, default='./models/dementia_cnn.pt')
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()

args = parse_args()

def main():

    image_transforms = transforms.Compose([
    transforms.CenterCrop(175),
    transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(args.training_image_directory, transform=image_transforms)

    batch_size = 128
    val_size = 1000
    train_size = len(dataset) - val_size 

    # train / test split
    train_data,val_data = random_split(dataset,[train_size,val_size])
    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Validation Data : {len(val_data)}")

    #load the train and validation into batches.
    train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True)

    # if available, move model to gpu for training else run on cpu
    if args.gpu and torch.backends.mps.is_available():
        print('moving model to MPS GPU')
        device = 'mps'
    else:
        print('GPU is not available. Training on CPU')
        device = 'cpu'
    model = AlzheimersMRIClassification(n_classes=4).to(device)

    # Model training
    opt_func = torch.optim.Adam
    lr = args.learning_rate # fitting the model on training data and record the result after each epoch
    history = fit(args.num_epochs, lr, model, train_dl, val_dl, opt_func, device=device)

    # save model state dict to output path
    torch.save(model.state_dict(), args.output_path)

if __name__ == '__main__':
    main()