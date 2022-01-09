# food-image-recognition

This repo contains code used to train the neural network classifier on the ifood-2019-fgvc6 dataset, The goal is to classify images into one of 251 food groups. The main challenges posed by the problem's creators are the fine-grained classes and noise in the data (mislabeled images, unrelated images, etc). 

<p float="left">
  <img src="/lucid-visualizations/class-1.png" width="200" />
  <img src="/lucid-visualizations/class-2.png" width="200" /> 
  <img src="/lucid-visualizations/class-3.png" width="200" />
</p>

## Tools Used 

* [numpy](https://numpy.org)
* [sklearn](https://scikit-learn.org/stable/)
* [matplotlib](https://matplotlib.org)
* [pytorch](https://pytorch.org)
* [lucid](https://github.com/tensorflow/lucid)

## Results

|           | parameters | Validation Loss | Validation accuracy  | Kaggle Score |
|-----------|------------|-----------------|----------------------|--------------|
| VGG11     | 129800187  | 4.71            | 25.45                | 0.437        |
| ResNet101 | 43014459   | 2.57            | 51.35                | 0.285        |
| ResNet152 | 58658107   | 1.49            | 53.13                | 0.273        |

## Acknowledgements 

* [pytorch-lucid](https://github.com/elichen/Feature- visualization/blob/master/Feature visualization.ipynb) 
* [residual-networks](https://towardsdatascience.com/hitchhikers-guide-to-residual-networks-resnet-in- keras-385ec01ec8ff) 
* [torch-implementations](https://pytorch.org/hub/pytorch_vision_resnet/) 
