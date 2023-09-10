# Project: Image Similarity Search

The purpose of this project is to perform a successful image similarity search, by training a siamese network using a triplet loss.

## Dataset Info:
The dataset used for this project is from HuggingFace - Matthijs/snacks (https://huggingface.co/datasets/Matthijs/snacks/tree/main)

Dataset contains 3 splits (train, validation, test) and 20 classifiers     \
train - 4838 images \
val - 955 images    \
test - 952 images   

## Siamese Model:

![Siamese Model Struct](https://github.com/parthshah231/image_similarity_search/blob/master/README/siamese_struct.png)

Siamese Network is a special type of neural net that is used extensively for learning object similarity, utilizing triplet loss as a loss to train the model. Triplet loss helps the network to learn the inputs by minimizing the distance between similar items and maximizing the distance between dissimilar items.

## Training and Validation curves
![Training Curves](https://github.com/parthshah231/image_similarity_search/blob/master/README/training_curves.png)

![Validation Curves](https://github.com/parthshah231/image_similarity_search/blob/master/README/validation_curves.png)

From the above plots:

For version 36 (backbone: "wide_resnet", "augment": false, epochs: 30, lr: 0.0003, wd: 0.01) \
Training accuracy: ~100% (1- train/triplet) \
Val accuracy: ~76% (1- val/triplet)

For version 37 (backbone: "wide_resnet", "augment": true, epochs: 30, lr: 0.0003, wd: 0.01) \
Training accuracy: ~98% (1- train/triplet) \
Val accuracy: ~86% (1- val/triplet)

Only difference between version 36 and version 37 is applying augments which helps in preventing overfitting and
generalizes faster in same time (30 epochs).

## Example outputs
![Output1](https://github.com/parthshah231/image_similarity_search/blob/master/README/output1.png)

![Output2](https://github.com/parthshah231/image_similarity_search/blob/master/README/output2.png)

![Output3](https://github.com/parthshah231/image_similarity_search/blob/master/README/output3.png)

## Usage:

Clone the repo:
```
git clone git@github.com:parthshah231/image_similarity_search.git
```

To train a new model you can either:
```
python train.py --backbone='efficientnet' --augment --epochs=30 --lr=3e-4 --wd=1e-2
```
or

1. Open ```run.py```
2. Define a grid

then,
```
python run.py
```

To evaluate the model:
```
python evaluate.py --version-number=30 --num-show-similar=8
```
Here, version number specifies a lightning model
(you can use any model that works best for you!). \
It will grab a config file present in the version folder and then load the trained model. After loading the model, it will select a random image from the test dataset and grab 'n' similar images from the test_dataset based on the similarity score. It will plot those images along with input image and similarity scores associated to the image.

## Further Implementations:
The best model still has some difficultly, as you can see from the 3rd example.
1. Implementing more augmentations to generalize the model such as TrivialAugmentWide which contains an array of augmentations.
2. Add a classfication task making it a multi-modal job and boost the classifiers which perform poorly (model has difficulty learning/classifying)
