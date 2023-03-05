import shap
import torch
import numpy as np
import matplotlib.pyplot as plt


class Explainer():

    def __init__(self, model, image_dl, class_names, background=None):

        self.class_names = class_names
        self.model = model

        test_images = []
        test_labels = []
        for batch in image_dl:
            images, labels = batch
            test_labels.append(labels)
            test_images.append(images.view(-1, 3, 175, 175))
        self.images = torch.cat(test_images, axis=0)
        L = len(self.images)
        if background == None:
            background = self.images[:int(L * 0.6)].mean(axis=0).unsqueeze(0)
        print(f'background shape: {background.shape}, test_images shape: {images.shape}')


        self.explainer = shap.DeepExplainer(model, background)

    def shap_values(self, images):
        return self.explainer.shap_values(images)

    def plot_shap_values(self, images):
        sv = self.shap_values(images)

        self.model.eval()
        preds = self.model.predict(images)
        probs = torch.sigmoid(self.model(images))

        labels = np.array([[str(c) for c in self.class_names] for i in range(len(images))])
        plot_labels = []
        for i, r in enumerate(labels):
            row = []
            for j, c in enumerate(r):
                row.append(f'{c} prob: {probs[i, j]*100:0.1f}%')
            plot_labels.append(row)
        plot_labels = np.array(plot_labels)

        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in sv]
        test_numpy = np.swapaxes(np.swapaxes(images.numpy(), 1, -1), 1, 2)
        print(shap_numpy[0].shape, test_numpy.shape)

        shap.image_plot(
            shap_numpy, 
            test_numpy, 
            labels=plot_labels, 
            true_labels=[f'true label: {p}' for p in list(preds.numpy())],
            show=False
            )

        fig = plt.gcf()
        fig.set_size_inches(18.5, 20)
        plt.tight_layout()
        plt.show()

        