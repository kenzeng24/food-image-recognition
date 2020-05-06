import _pickle as pickle
import numpy as np
import os
import pandas as pd 
from skimage.io import ImageCollection, imshow, concatenate_images
from skimage.transform import resize

#README:
#The functions take in the desired size of the image,
#but the returned array is 2 dimensional. For example, if dim=(256,256),
#then each example will be an array of size 196608

def load_ifood(im_dir, label_dir, dim):
    # TODO: joblib to parallelize
    # https://scikit-image.org/docs/dev/user_guide/tutorial_parallelization.html
    images = ImageCollection(im_dir)
    images_resized = [resize(image, dim) for image in images]
    num_images = len(images_resized)
    image_arr = np.reshape(concatenate_images(images_resized), (num_images, -1))

    labels = pd.read_csv(label_dir)
    labels = labels.sort_values(by=['img_name'])
    labels = labels.head(num_images)
    y = labels["label"].to_numpy()
    unique_labels = labels["label"].unique()
    return image_arr, y, unique_labels

def partial_train(dim=(256, 256)):
    return load_ifood('train_partial/*.jpg', 'train_labels.csv', dim)

user_path = "/Users/kenzeng/Desktop/College/COMP/COMP540/Project/ifood-2019-fgvc6/"

def load_train(dim=(256, 256), path=user_path):
    return load_ifood(path + 'train_set/*.jpg',
                      path + 'train_labels.csv', dim)

def load_val(dim=(256, 256), path=user_path):
    return load_ifood(path + 'val_set/*.jpg',
                      path + 'val_labels.csv', dim)

def load_test(dim=(256, 256), path=user_path):
    return load_ifood(path + 'test_set/*.jpg',
                      path + 'test_labels.csv', dim)

def load_all(path=user_path,
             dim=(256, 256)):
    X_train, y_train, _ = load_train(dim=dim, path=path)
    X_val, y_val, _ = load_val(dim=dim, path=path)
    X_test, y_test, _ = load_test(dim=dim, path=path)
    return X_train, y_train, X_val, y_val, X_test, y_test

# if __name__ == "__main__":
#     X_test, y_test, _ = load_test()
#     print(X_test.shape)
#     print(y_test.shape)
