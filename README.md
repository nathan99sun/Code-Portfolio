# MIT-Maker-Portfolio

Welcome to my MIT Maker Portfolio!

I am an avid computer programmer, and throughout high school, I've delved into machine learning and strived to utilize computer science in my math research.

The following gives a brief description of each of the codes:

## Generative adversarial networks (GANs)

GANs, or generative adversarial networks, contain two neural networks that compete with each other and improve at the same time. Eventually, they enhance themselves so much that even humans cannot distinguish the images they produce with real images. Here, I've trained my GAN to generate realistic handwritten digits and pictures of animals/objects. I used the MNIST, CIFAR10, and CIFAR100 datasets.

DRAGAN-main is the main file to run. 

dataloader and utils are used to help DRAGAN and WassersteinGAN run.

DRAGAN is an implementation of the Deep Regret Analytic Generative Adversarial Network, recently developed by Kodali et al to mitigate the effects of mode collapse. They hypothesize that mode collapse is due to the min-max game converging to a bad local equilibria in non-entirely convex games. To counter this effect, the paper suggests the use of gradient penalty points through a gradient penalty scheme, leading to the creation of DRAGAN.

WassersteinGAN is an implementation of the Wasserstein GAN, which improves stability compared to the regular GANs. It uses Earth mover's distance to create a new loss function, which improves the quality of generated images.

## Data!

Enjoy some cool gifs of the DRAGAN and WGAN training on the MNIST, CIFAR10, and CIFAR100 datasets!

## PROMYS q-Eulerian Polynomial Data Generator

eulerian: This project was made in PROMYS 2018 to generate data for q-Eulerian polynomials proposed by Paul Gunnells (https://arxiv.org/pdf/1702.02446.pdf). See https://arxiv.org/pdf/1809.07398.pdf Definition 2.3, 2.6 for algorithm, also Appendix A for data. As there is no previously known efficient way to generate q-Eulerian polynomials, this program is able to generate data for n=10 with given data restraints, allowing us to make important conjectures about the properties of these polynomials and ultimately allowing us to derive a new formula for these polynomials.

## CIFAR100 Neural Network

CIFARtest is a Convolutional Neural Network trained on the CIFAR100 dataset. Final commented accuracies are ran on 10 epochs each.

IMPORTANT: when running these three programs, change the paths, as they save the best model for test and validation.
