<p float="left">
  <img src="/lucid-visualizations/class-1.png" width="200" />
  <img src="/lucid-visualizations/class-2.png" width="200" /> 
  <img src="/lucid-visualizations/class-3.png" width="200" />
</p>

# food-image-recognition

Automatic food classification can be used to develop applications that help people monitor their food intake and lead healthier lives. However, this task is challenging problem due to the shear variety of different food groups and the similarity between groups.This repo contains code used to train the neural network classifier on the ifood-2019-fgvc6 dataset, The goal is to classify images into one of 251 food groups. The main challenges posed by the problem's creators are the fine-grained classes and noise in the data (mislabeled images, unrelated images, etc).  

## Tools Used 

* [numpy](https://numpy.org)
* [sklearn](https://scikit-learn.org/stable/)
* [matplotlib](https://matplotlib.org)
* [pytorch](https://pytorch.org)
* [lucid](https://github.com/tensorflow/lucid)

## Methods

Our final model used the ResNet 152 architecture. We chose this architecture because it performed the best as we trained various networks for two epochs (described in section 3). It performed the best in that it had the smallest validation loss and highest accuracy of the models we tested. We also chose this architecture over VGGNet due to concerns over training time and computational expense. VGG-11 took longer to train on one of our machines than ResNet-152, and VGG-11 was the smallest version of VGGNet. To best implement VGGNet, we would need to train a larger version (such as VGG-16 or VGG-19), which could take days to train with a sufficient number of epochs one time.\par

The models weights are initialized to weights pre-trained for the ImageNet data set and provided by PyTorch. This reduced training time substantially and yielded higher accuracies much faster than other initialization methods. The reasoning behind this success is that the first few convolutional layers in deep CNNs tend to correspond to simple features (curves, lines, etc.) that are informative in many different image classification problems.

## Results

While our model's accuracy could have been higher, we were satisfied and impressed with the speed it achieved those results and how simple libraries such as PyTorch made implementing it. Our Kaggle scores indicated that around 83\% of the time, the correct class class of a testing image would be in our model's top 3. This high top-3 accuracy assures us our methods were effective and could potentially produce an extremely accurate model

|           | parameters | Validation Loss | Validation accuracy  | Kaggle Score |
|-----------|------------|-----------------|----------------------|--------------|
| VGG11     | 129800187  | 4.71            | 25.45                | 0.437        |
| ResNet101 | 43014459   | 2.57            | 51.35                | 0.285        |
| ResNet152 | 58658107   | 1.49            | 53.13                | 0.273        |

## Acknowledgements 

* [pytorch-lucid](https://github.com/elichen/Feature-visualization) 
* [residual-networks](https://towardsdatascience.com/hitchhikers-guide-to-residual-networks-resnet-in-keras-385ec01ec8ff) 
* [torch-implementations](https://pytorch.org/hub/pytorch_vision_resnet/) 
