from typing import Tuple
import datetime

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Args:
    """TODO: Command-line arguments to store model configuration.
    """
    num_classes = 10

    # Hyperparameters
    epochs = 25     # Should easily reach above 65% test acc after 20 epochs with an hidden_size of 64
    batch_size = 15
    lr = 1e-3
    weight_decay = 1e-3

    # TODO: Hyperparameters for ViT
    # Adjust as you see fit
    input_resolution = 32
    in_channels = 3
    patch_size = 4
    hidden_size = 64
    layers = 6
    heads = 8

    # Save your model as "vit-cifar10-{YOUR_CCID}"
    YOUR_CCID = "asunil"
    name = f"vit-cifar10-{YOUR_CCID}"

class PatchEmbeddings(nn.Module):
    """TODO: (0.5 out of 10) Compute patch embedding
    of shape `(batch_size, seq_length, hidden_size)`.
    """
    def __init__(
        self, 
        input_resolution: int,
        patch_size: int,
        hidden_size: int,
        in_channels: int = 3,      # 3 for RGB, 1 for Grayscale
        ):
        super().__init__()
        # #########################
        # Finish Your Code HERE
        # #########################

        #Number of patches first:
        self.num_patches = (input_resolution//patch_size) ** 2

        #Convolution layer
        self.projection = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size, #Each kernel is the size of the patch
            stride=patch_size #to not have overlapping patches
        )
        
        # #########################

    def forward(
        self, 
        x: torch.Tensor,
        ) -> torch.Tensor:
        # #########################
        # Finish Your Code HERE
        # #########################

        x = self.projection(x)

        #Flatten
        x = x.flatten(2)
        embeddings = x.transpose(1,2)

        # #########################
        return embeddings

class PositionEmbedding(nn.Module):
    def __init__(
        self,
        num_patches: int,
        hidden_size: int,
        ):
        """TODO: (0.5 out of 10) Given patch embeddings, 
        calculate position embeddings with [CLS] and [POS].
        """
        super().__init__()
        # #########################
        # Finish Your Code HERE
        # #########################

        self.cls_token = nn.Parameter(torch.zeros(1,1 , hidden_size)) #(batch_size, CLS token, size of feature_vector)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size)) #(batch_size, +1 for CLS token, size of feature_vector)

        # #########################

    def forward(
        self,
        embeddings: torch.Tensor
        ) -> torch.Tensor:
        # #########################
        # Finish Your Code HERE
        # #########################

        batch_size = embeddings.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1) #batch_size changed from default.

        embeddings = torch.cat((cls_tokens, embeddings), dim = 1)

        embeddings += self.position_embeddings #Adding position data to the features


        # #########################
        return embeddings


class TransformerEncoderBlock(nn.Module):
    """TODO: (0.5 out of 10) A residual Transformer encoder block.
    """
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        # #########################
        # Finish Your Code HERE
        # #########################

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head)
        self.ln_1 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), #increases out_dimension by 4 times.
            nn.GELU(),
            nn.Linear(4 * d_model, d_model) #Reverts the dimension change after linear activation GELU.
        )
        self.ln_2 = nn.LayerNorm(d_model, elementwise_affine=False) #Since CIFAR10 is relatively prone to overfitting by transformer architecture, turn off affine parameters

        # #########################

    def forward(self, x: torch.Tensor):
        # #########################
        # Finish Your Code HERE
        # #########################
        attn_output, attn_weights = self.attn(x, x, x)
        x = x + attn_output
        x = self.ln_1(x)

        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.ln_2(x)

        # #########################
    

        return x


class ViT(nn.Module):
    """TODO: (0.5 out of 10) Vision Transformer.
    """
    def __init__(
        self, 
        num_classes: int,
        input_resolution: int, 
        patch_size: int, 
        in_channels: int,
        hidden_size: int, 
        layers: int, 
        heads: int,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        # #########################
        # Finish Your Code HERE
        # #########################
        self.patch_embed = PatchEmbeddings(input_resolution, patch_size, hidden_size, in_channels)

        self.pos_embed = PositionEmbedding((input_resolution//patch_size)**2, hidden_size)

        self.ln_pre = nn.LayerNorm(hidden_size)

        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(d_model=hidden_size, n_head=heads)
            for _ in range(layers)
        ])

        self.ln_post = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # #########################


    def forward(self, x: torch.Tensor):
        # #########################
        # Finish Your Code HERE
        # #########################

        #first we make patch embeddings
        x = self.patch_embed(x)

        #add position embeddings
        x = self.pos_embed(x)

        #Pre-Norm
        x = self.ln_pre(x)

        #Encoder block
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)

        x = self.ln_post(x)

        cls_token_embedding = x[:, 0] #Extract our class token

        x = self.classifier(cls_token_embedding) #Pass our token to classifier

        # #########################

        return x


def transform(
    input_resolution: int,
    mode: str = "train",
    mean: Tuple[float] = (0.5, 0.5, 0.5),   # NOTE: Modify this as you see fit
    std: Tuple[float] = (0.5, 0.5, 0.5),    # NOTE: Modify this as you see fit
    ):
    """TODO: (0.25 out of 10) Preprocess the image inputs
    with at least 3 data augmentation for training.
    """
    if mode == "train":
        # #########################
        # Finish Your Code HERE
        # #########################
        tfm = transforms.Compose([
            transforms.RandomResizedCrop(input_resolution), #Augment 1
            transforms.RandomHorizontalFlip(), #Augment 2
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), #Augment 3
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        # #########################

    else:
        # #########################
        # Finish Your Code HERE
        # #########################
        # Preprocessing for validation/test
        tfm = transforms.Compose([
            transforms.Resize(input_resolution),
            transforms.CenterCrop(input_resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        # #########################

    return tfm

def inverse_transform(
    img_tensor: torch.Tensor,
    mean: Tuple[float] = (-0.5/0.5, -0.5/0.5, -0.5/0.5),    # NOTE: Modify this as you see fit
    std: Tuple[float] = (1/0.5, 1/0.5, 1/0.5),              # NOTE: Modify this as you see fit
    ) -> np.ndarray:
    """Given a preprocessed image tensor, revert the normalization process and
    convert the tensor back to a numpy image.
    """
    # #########################
    # Finish Your Code HERE
    # #########################
    inv_normalize = transforms.Normalize(mean=mean, std=std)
    img_tensor = inv_normalize(img_tensor).permute(1, 2, 0)
    img = np.uint8(255 * img_tensor.numpy())
    # #########################
    return img


def train_vit_model(args):
    """TODO: (0.25 out of 10) Train loop for ViT model.
    """
    # #########################
    # Finish Your Code HERE
    # #########################
    # -----
    # Dataset for train / test
    tfm_train = transform(
        input_resolution=args.input_resolution, 
        mode="train",
    )

    tfm_test = transform(
        input_resolution=args.input_resolution, 
        mode="test",
    )

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tfm_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tfm_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # -----
    # TODO: Define ViT model here
    model = ViT(
        num_classes=args.num_classes,
        input_resolution=args.input_resolution,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads
    )
    # print(model)

    if torch.cuda.is_available():
        model.cuda()

    # TODO: Define loss, optimizer and lr scheduler here
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.lr, gamma = args.weight_decay)

    # #########################
    # Evaluate at the end of each epoch
    best_acc = 0.0
    for epoch in range(args.epochs):
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} / {args.epochs}")

        for i, (x, labels) in enumerate(pbar):
            model.train()
            # #########################
            # Finish Your Code HERE
            # #########################
            optimizer.zero_grad()
            if torch.cuda.is_available():
                x, labels = x.cuda(), labels.cuda()

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, labels)

            #Backpass
            loss.backward()
            optimizer.step()

            # #########################

            # NOTE: Show train loss at the end of epoch
            # Feel free to modify this to log more steps
            pbar.set_postfix({'loss': '{:.4f}'.format(loss.item())})

        scheduler.step() #Take a step on LR adjustment.
        # Evaluate at the end
        test_acc = test_classification_model(model, test_loader)

        # NOTE: DO NOT CHANGE
        # Save the model
        if test_acc > best_acc:
            best_acc = test_acc
            state_dict = {
                "model": model.state_dict(),
                "acc": best_acc,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            torch.save(state_dict, "{}.pt".format(args.name))
            print("Best test acc:", best_acc)
        else:
            print("Test acc:", test_acc)
        print()

def test_classification_model(
    model: nn.Module,
    test_loader,
    ):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
