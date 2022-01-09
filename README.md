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
