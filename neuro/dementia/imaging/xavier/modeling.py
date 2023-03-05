import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np


def accuracy(outputs, labels):
    # this method computes the accuracy score
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

  
@torch.no_grad()
def evaluate(model, val_loader):
    # this method computes validation loss
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

  
def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD, device='cpu'):
    # this method trains the convolutional neural network
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch, device)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)


def save_checkpoint(model, path=None, optim=None, epoch=None, loss=None):
    # this method saves the torch model state dict to a specified directory.
    # if path is not given, the model is saved to ./checkpoints/
    #
    # can optionally save the optmizer, epoch and loss as well
    checkpoint = {'model_state_dict' : model.state_dict()}
    if optim:
        checkpoint['optim_state_dict'] = optim
    if epoch:
        checkpoint['EPOCH'] = epoch
    if loss:
        checkpoint['LOSS'] = loss

    if not path:
        path = f'./checkpoints/xavier_state_dict_{epoch}.pt'
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer):
    # this method loads a torch model from a path as saved by the above function.
    #
    # here, the model argument is an initialied untrained model object.
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch, device):
        images, labels = batch
        images, labels = images.to(device), labels.to(device) 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

class DementiaMRIClassification(ImageClassificationBase):


    def __init__(self, n_classes, transforms=None, image_size=(175,175), n_channels=3, p=0.2, conv_layers=[(32, 3), (64, 3), (128, 3), (256, 3)], linear_layers=[1024, 512]):
        '''
        This implementation takes in the following additional arguments:

        conv_layers: a list of tuples specifying the number of output channels and kernel size for each 
                     convolutional layer

        linear_layers: a list specifying the size of each linear layer

        The _get_flattened_size method calculates the size of the flattened output of the convolutional layers, 
        which is needed to construct the linear layers.

        Note that this implementation assumes that all convolutional layers have the same stride of 1 and padding that maintains the input size.
        '''
        
        super().__init__()

        self.transforms = transforms
        self.n_conv_layers = len(conv_layers)
        self.n_channels = n_channels
        self.image_size = image_size

        # convolutional layers
        conv_blocks = []
        in_channels = n_channels
        for i, (out_channels, kernel_size) in enumerate(conv_layers):
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            conv_blocks.append(conv_block)
            in_channels = out_channels
        self.conv_blocks = nn.ModuleList(conv_blocks)

        # linear layers
        in_features = self._get_flattened_size()
        linear_blocks = []
        for i, out_features in enumerate(linear_layers):
            if i == 0:
                linear_block = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                )
            else:
                linear_block = nn.Sequential(
                    nn.Linear(linear_layers[i-1], out_features),
                    nn.ReLU(),
                    nn.Dropout(p=p)
                )
            linear_blocks.append(linear_block)
            in_features = out_features
        linear_blocks.append(nn.Linear(in_features, n_classes))
        self.linear_blocks = nn.ModuleList(linear_blocks)

    def forward(self, x):
        if self.transforms is not None:
            x = self.transforms(x)
        for i in range(self.n_conv_layers):
            x = self.conv_blocks[i](x)
        x = x.view(x.size(0), -1)
        for i in range(len(self.linear_blocks)):
            x = self.linear_blocks[i](x)
        return x

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, self.n_channels, self.image_size[0], self.image_size[1])
            for i in range(self.n_conv_layers):
                x = self.conv_blocks[i](x)
            return x.view(1, -1).size(1)

    def predict(self, x):
        with torch.no_grad():
            hats = self.forward(x)
        _, preds = torch.max(hats, dim=1)
        return preds

    def _get_conv_layers(self):
        l = [module for module in self.modules() if not isinstance(module, nn.Sequential)][1:]
        weights = []
        conv_layers = []
        for layer in l:
            if type(layer) == nn.Conv2d:
                weights.append(layer.weight)
                conv_layers.append(layer)
        return weights, conv_layers

    def _single_image_feature_map(self, image, with_activation=False):
        '''Expects RGB PIL Image'''
        transformed_image = self.transforms(image).unsqueeze(0)
        _, conv_layers = self._get_conv_layers()
        outputs = []
        names = []
        feature_image = transformed_image
        for layer in conv_layers:
            feature_image = layer(feature_image)
            if with_activation:
                feature_image = nn.ReLU()(feature_image)
            outputs.append(feature_image)
            names.append(str(layer))  

        processed = []
        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map,0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())

        return names, processed
    
    def plot_feature_maps(self, names, processed=None, image=None, ):
        if processed == None:
            processed = self._single_image_feature_map(image, with_activation=True)
        fig = plt.figure(figsize=(120, 80))
        for i in range(len(processed)):
            a = fig.add_subplot(5, 4, i+1)
            imgplot = plt.imshow(processed[i])
            a.axis("off")
            a.set_title(names[i], fontsize=30)

    def average_activation(self, dl, mode='layers'):
        batch_activation = []
        for batch in dl:
            images, _ = batch
            for image in images:
                image_pil = T.ToPILImage()(image.squeeze())
                _, processed = self._single_image_feature_map(image_pil, with_activation=True)
                batch_activation.append(processed)
        if mode == 'layers':
            return np.array(batch_activation).mean(axis=1)
        elif mode == 'images':
            return np.array(batch_activation).mean(axis=0)
        else:
            raise NotImplementedError