# 1. What is Convolution?

## 1.1 Understanding Convolution from a Mathematical Perspective

The mathematical formula for convolution is typically represented as the convolution of two functions, \( f(x) \) and \( g(x) \). It is defined as:

$$
(f * g)(x) = \int_{-\infty}^{\infty} f(t) \cdot g(x - t) \, dt
$$

Where:

- \( f(x) \) and \( g(x) \) are the two functions to be convolved.
- \( (f * g)(x) \) is the resulting function after the convolution.
- \( t \) is the integration variable.

For discrete convolution, the formula is:

$$
(f * g)[n] = \sum_{k=-\infty}^{\infty} f[k] \cdot g[n - k]
$$

Here, \( f[k] \) and \( g[k] \) are discrete signals, and \( n \) is the discrete output index.

## 1.2 Visualizing Convolution

In the image, the left part shows the pixel value matrix of a grayscale image (i.e., how the image is represented as numbers for a computer). In the middle is the **convolution kernel** matrix, which slides over the original image starting from the top-left corner. The kernel computes a value at each position, and this process is repeated across the image. The resulting values form the **right image (feature map)**, which contains the local features of the original image obtained through the convolution process.

![1](https://github.com/user-attachments/assets/53f5b817-0e70-40c6-9294-d3ec78b2b2c2)

The animation works like this: 
![2](https://github.com/user-attachments/assets/f7697387-d053-4128-83b2-7ec005072e44)

# 2. Diffusion Model Principles

## 2.1 Principles of Early Generative Models

Early generative models such as GAN (Generative Adversarial Networks) and VAE (Variational Autoencoders) involve inverting the original model. For instance, in GAN, the recognition model is a conventional convolution network used for identifying generated images. The generation model, however, reverses this by using a transposed convolution network (also called deconvolution) to generate images, but this approach doesn't produce ideal results.

Let's talk about transposed convolution. Transposed convolution is the reverse of convolution: convolution turns a large matrix into a smaller one, whereas transposed convolution takes a smaller matrix and generates a larger one. As shown in the image below, it creates the dashed-line matrix!
![3](https://github.com/user-attachments/assets/83fbddff-10fd-4fa5-9bc0-6e21785d1407)


## 2.2 Diffusion Model

While directly generating images is not ideal, scientists have drawn inspiration from diffusion in physics. In nature, substances tend to move toward a disordered state. For example, when a drop of ink is added to a glass of water, it gradually spreads out. This suggests that generative models could also take a gradual, step-by-step approach instead of rushing, aiming for steady progress.

Thus, diffusion models were born. We start by adding noise to an image’s pixels, which results in a very chaotic image. Conversely, we can also reverse this process to recover the original image from this noisy one.

![4](https://github.com/user-attachments/assets/9a1ed29c-9b05-480b-abe6-77b4268c9429)

## 2.3 Convolution in Diffusion Models

Diffusion models typically use a UNet network to predict denoised images, with the addition of **timestep** to reflect the noise level. The prediction is done for each **timestep** of the image.

As shown in the image, this is a convolution kernel from the UNet network used in diffusion (with code implementation to follow). In fact, throughout the entire network, the properties of the convolution kernel remain largely unchanged, and the input's width and height do not change during the forward pass. Only the number of channels changes.

![5](https://github.com/user-attachments/assets/a8f12985-c9c2-4776-958f-a020387357f1)
We remembered, **convolution** maps a matrix onto a feature matrix, while **diffusion** introduces disorder into the matrix.Try to think of it this way: **convolution** disturbs or restores the local features of a matrix, while **diffusion** relies on **convolution** to diffuse local features.

![6](https://github.com/user-attachments/assets/7b937214-fa6f-426f-9037-4094e9f5e578)



# 3. Code Implementation of Diffusion

Theory is one thing, but let’s dive into a practical example.

## 3.1 Importing Required Libraries

```python
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## 3.2 Using the MNIST Dataset

```python
dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor()
)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

## 3.3 Writing the Noise Corruption Formula

Corrupting means mixing the image with noise in a certain proportion to achieve denoising. As the diffusion process progresses, the image becomes clearer, and the noise has less of an effect.

```python
def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Adjust shape for broadcasting
    return x * (1 - amount) + noise * amount
```

## 3.4 Creating a Simple UNet Model

We’ll use a mini UNet model (not the standard one) that still achieves good results.

```python
class BasicUNet(nn.Module):
    """A minimal UNet implementation."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.act = nn.SiLU()  # Activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Pass through the layer and activation function
            if i < 2:  # Skip connection for all but the final down layer
                h.append(x)  # Store output for skip connection
                x = self.downscale(x)  # Downscale for next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all but the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Use stored output (skip connection)
            x = self.act(l(x))  # Pass through the layer and activation function

        return x
```

The network looks like this:

![7](https://github.com/user-attachments/assets/b6fa8e41-f15b-477b-a9f1-59cadc9e3770)


## 3.5 Defining the Training Parameters

```python
# Dataloader (adjust batch size)
batch_size = 128
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Number of epochs
n_epochs = 30

# Create the network
net = UNet2DModel(
    sample_size=28,  # Target image resolution
    in_channels=1,  # Input channels, 3 for RGB images
    out_channels=1,  # Output channels
    layers_per_block=2,  # ResNet layers per UNet block
    block_out_channels=(32, 64, 64),  # Matching our basic UNet example
    down_block_types=(
        "DownBlock2D",  # Regular ResNet downsampling block
        "AttnDownBlock2D",  # ResNet block with attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # ResNet block with attention
        "UpBlock2D",  # Regular ResNet upsampling block
    ),
)
net.to(device)

# Loss function
loss_fn = nn.MSELoss()

# Optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# Track losses
losses = []

# Training loop
for epoch in range(n_epochs):
    for x, y in train_dataloader:
        x = x.to(device)  # Data on GPU
        noise_amount = torch.rand(x.shape[0]).to(device)  # Random noise amount
        noisy_x = corrupt(x, noise_amount)  # Create noisy input

        # Get model prediction
        pred = net(noisy_x, 0).sample  # Use timestep 0

        # Calculate the loss
        loss = loss_fn(pred, x)  # Compare to original clean image

        # Backprop and update parameters
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
    print(f"Epoch {epoch}. Average loss: {avg_loss:.5f}")
```

## 3.6 Testing the Model

After training, we can test the diffusion model’s power:

```python
n_steps = 40
x = torch.rand(8, 1, 28, 28).to(device)  # Start from random noise
step_history = [x.detach().cpu()]
pred_output_history = []

for i in range(n_steps):
    with torch.no_grad():  # No gradients during inference
        pred = net(x,0).sample  # Predict denoised image
    pred_output_history.append(pred.detach().cpu())  # Store prediction
    mix_factor = 1 / (n_steps - i)  # Mix towards prediction
    x = x * (1 - mix_factor) + pred * mix_factor  # Move partway to the denoised image
    step_history.append(x.detach().cpu())  # Store for plotting

fig, axs = plt.subplots(n_steps, 2, figsize=(9, 32), sharex=True)
axs[0, 0].set_title("Input (noisy)")
axs[0, 1].set_title("Model Prediction")
for i in range(n_steps):
    axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap="Greys")
    axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1), cmap="Greys")
```

![output](https://github.com/user-attachments/assets/2426e5b6-d33e-49b9-acdd-465e1c7a634d)

This shows the final results of testing the model, and the output looks quite impressive!

# Conclusion

This concludes the explanation of the diffusion model. You can try modifying the UNet network or adjusting parameters to see if you can achieve even more remarkable results!