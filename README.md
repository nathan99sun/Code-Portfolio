# MIT-Maker-Portfolio

Welcome to my MIT Maker Portfolio!

I am an avid computer programmer, and throughout high school, I've delved into machine learning and strived to utilize computer science in my math research. Computers can accomplish stunning feats that humans alone could not even dream of, and that never ceases to amaze me. So I've started to apply computer science to other fields and explored topics that sparked my curiosity, in hope that one day I could create something that would benefit society.

The following gives a brief description of each of the codes and files:

## Generative adversarial networks (GANs)

GANs, or generative adversarial networks, contain two neural networks that compete with each other and improve at the same time. Eventually, they enhance themselves so much that even humans cannot distinguish the images they produce with real images. Here, I've trained a DRAGAN and a WassersteinGAN to generate realistic handwritten digits and pictures of animals/objects. I used the MNIST, CIFAR10, and CIFAR100 datasets.

A few technical notes:

DRAGAN-main is the main file to run. 

dataloader and utils are used to help DRAGAN and WassersteinGAN run.

DRAGAN is an implementation of the Deep Regret Analytic Generative Adversarial Network, recently developed by Kodali et al to mitigate the effects of mode collapse. They hypothesize that mode collapse is due to the min-max game converging to a bad local equilibria in non-entirely convex games. To counter this effect, the paper suggests the use of a gradient penalty scheme, leading to the creation of DRAGAN.

WGAN is an implementation of the Wasserstein GAN, which improves stability compared to a classical GAN. It uses Earth mover's distance to create a new loss function, improving the quality of generated images.

# Data!!!

Enjoy some cool gifs of the DRAGAN and Wasserstein GAN training on the MNIST, CIFAR10, and CIFAR100 datasets!

## PROMYS q-Eulerian Polynomial Data Generator

eulerian: I made this project in PROMYS 2018 to generate data for q-Eulerian polynomials proposed by Paul Gunnells (https://arxiv.org/pdf/1702.02446.pdf). See https://arxiv.org/pdf/1809.07398.pdf Definition 2.3, 2.6 for algorithm, also Appendix A for data. As there is no previously known efficient way to generate q-Eulerian polynomials, this program is able to generate data for n=10 with given data restraints, allowing us to make important conjectures about the properties of these polynomials and ultimately allowing us to derive a new formula for these polynomials.

## CIFAR100 Neural Network


CIFARtest is a Convolutional Neural Network trained on the CIFAR100 dataset. It is ideal for image recognition, and it tries to seperate the CIFAR100 images into 100 catagories. Final commented accuracies are ran on 10 epochs each.

IMPORTANT: when running these three programs, change the paths, as they save the best model for test and validation.
