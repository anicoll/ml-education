---
sidebar_position: 1
title: Convolutional networks (CNNs)
---

# Convolutional networks (CNNs)

An MLP treats an image as a flat list of pixels. It has no idea that neighbouring pixels are related, or that a cat in the top-left corner is the same as a cat in the bottom-right. CNNs build those assumptions in, and that's why they dominated computer vision for a decade.

## The convolution

A **filter** (or kernel) is a small grid of weights, say 3×3. Slide it across the image; at each position, take the dot product between the filter and the patch of pixels underneath. The result is a **feature map** showing where in the image the filter's pattern appears.

```text
Image patch      Filter (vertical edge)     Output
[ 1  1  0 ]      [ 1  0 -1 ]
[ 1  1  0 ]  ·   [ 1  0 -1 ]   =   3   (strong edge here)
[ 1  1  0 ]      [ 1  0 -1 ]
```

Two key ideas:

- **Local connectivity:** each output only looks at a small neighbourhood.
- **Weight sharing:** the same filter is used everywhere, so a pattern is detected wherever it appears, and the layer has very few parameters (a 3×3 filter on a colour image has 27 weights plus a bias).

A conv layer learns many filters at once (32, 64, 256...), producing that many feature maps, called **channels**.

## Pooling and stride

To build up from small patterns to big ones, the network reduces spatial size as it goes deeper:

- **Max pooling** takes the maximum over each 2×2 block, halving width and height.
- **Stride 2** convolutions skip every other position, with the same effect.

Each layer then "sees" a larger region of the original image (its **receptive field**). Early layers detect edges; later layers combine edges into textures, then parts, then objects.

## A small CNN in PyTorch

```python
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (3, 32, 32) -> (32, 32, 32)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (32, 16, 16)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (64, 8, 8)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                      # -> (128, 1, 1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
```

## Landmark architectures

| Model | Year | Contribution |
| --- | --- | --- |
| LeNet | 1998 | Early CNN for handwritten digits. |
| AlexNet | 2012 | Deep CNN on GPUs, won ImageNet by a huge margin. Started the deep learning boom. |
| VGG | 2014 | Showed that deeper stacks of small 3×3 filters work well. |
| ResNet | 2015 | Residual connections, enabling 100+ layer networks. |
| Vision Transformer (ViT) | 2020 | Showed transformers can match CNNs on images, given enough data. |

## Transfer learning

You rarely train a vision model from scratch. Instead, take a network pre-trained on ImageNet, replace its final layer, and fine-tune on your own images. A few hundred examples per class can be enough. This is the same idea as fine-tuning LLMs in Part 6, and it's a great first taste of it.

```python
import torchvision
model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

:::tip[Try it]
1. Train `SmallCNN` on CIFAR-10 (`torchvision.datasets.CIFAR10`) on your GPU. Aim for about 80% test accuracy. Add random crops and horizontal flips as data augmentation and see how much it helps.
2. Fine-tune a pre-trained ResNet-18 on the same data and compare.
3. Visualise the 32 filters of the first conv layer as tiny images. Can you see edge detectors?
:::

## Check your understanding

<MultipleChoice
  id="cnn-1"
  question="What does weight sharing in a convolution layer mean?"
  options={[
    "All layers use the same weights",
    "The same small filter is slid across the whole image, so a pattern is detected wherever it appears, with very few parameters",
    "Weights are shared between the training and test sets",
    "Every pixel gets its own weight",
  ]}
  answer={1}
  explanation="A 3×3 filter on a colour image has just 27 weights plus a bias, reused at every position. An MLP would need a separate weight for every pixel."
/>

<MultipleChoice
  id="cnn-2"
  question="A 32×32 feature map goes through a 2×2 max pooling layer. What size is the output?"
  options={["32×32", "16×16", "8×8", "64×64"]}
  answer={1}
  explanation="Max pooling over 2×2 blocks halves the width and height, so each later layer 'sees' a bigger region of the original image."
/>

<MultipleChoice
  id="cnn-3"
  question="You have 300 photos each of 5 kinds of plant. What's the most practical way to build a good classifier?"
  options={[
    "Train a large CNN from scratch",
    "Transfer learning: take a network pre-trained on ImageNet, replace its last layer, and fine-tune",
    "Use linear regression on the raw pixels",
    "Collect a million more photos first",
  ]}
  answer={1}
  explanation="A pre-trained network already knows edges, textures and shapes. With a new final layer, a few hundred examples per class is often enough."
/>

<OpenQuestion
  id="cnn-4"
  question="Why does a CNN work better than an MLP on images? Give two reasons.">

1. **It knows neighbouring pixels belong together.** Each filter looks at a small local patch, which matches how images work: edges and textures are local. An MLP treats the image as a flat list and has to discover from scratch that pixel 37 is next to pixel 38.
2. **It recognises a pattern anywhere.** Because the same filter slides over the whole image (weight sharing), a cat's ear detected in the top-left uses the same weights as one in the bottom-right. An MLP would have to learn separate weights for every position.

A bonus third reason: far fewer parameters. Fewer parameters means less overfitting and less data needed.

</OpenQuestion>
