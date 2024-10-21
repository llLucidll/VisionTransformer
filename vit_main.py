import torchvision
import numpy as np
import matplotlib.pyplot as plt

from vit_submission import *

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def inspect_data(
    transform: callable,
    n_imgs: int = 5,
    ):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    indices = np.random.randint(0, len(dataset), size=(n_imgs, ))
    # Visualize with matplotlib
    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        img = inverse_transform(img_tensor)
        plt.subplot(1, n_imgs, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(classes[label])

    plt.show()
    del dataset


def main():
    args = Args()
    try:
        import paramparse
        paramparse.process(args)
    except ImportError:
        print("WARNING: You have not installed paramparse. Please manually edit the arguments.")

    # -----
    # NOTE: Always inspect your data
    inspect_data(transform(
        input_resolution=args.input_resolution, 
        mode="train",
    ))

    # -----
    # TODO: Train your ViT model
    train_vit_model(args)


if __name__ == "__main__":
    main()