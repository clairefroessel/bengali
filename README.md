# Bengali.AI

<img src="./images/intro_pic.png" width="400" height="400" />

### Table of Contents

- [Description](#description)
- [How To Use](#how-to-use)
- [Resources](#resources)
- [License](#license)

---

## Description

This project was made for a Kaggle competition held in 2020. 

We are given the image of a handwritten Bengali grapheme and are challenged to separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.

I used a Residual Network (ResNet-50). The only change I made to the below architecture is deviding the final layer into 3 to answer to that challenge.

<img src="./images/resnet50.png" width="600" height="400" />

Also you can see in this repository how to do data augmentation using Keras, such as cutmix.

The performance of the model is 92% of recall (hierarchical macro-averaged - this is the metric used for the competition you can check here for more information : https://www.kaggle.com/c/bengaliai-cv19/overview/evaluation).

## Data

All the data are available here : https://www.kaggle.com/c/bengaliai-cv19/data

 There are roughly 10,000 possible graphemes, of which roughly 1,000 are represented in the training set. The test set includes some graphemes that do not exist in train but has no new grapheme components.

## How to use

This code works with Python >3.4, although it requires some additional packages such as Pandas, Numpy, opencv etc. I recommend using Anaconda python, and conda to install these.

---

## Resources

- [Deep Residual Learning for Image Recognition ](https://arxiv.org/abs/1512.03385)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899)
- [Image preprocessing crop and resize](https://www.kaggle.com/iafoss/image-preprocessing-128x128)
---

## License

MIT License

Copyright (c) [2017] [James Q Quick]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[Back To The Top](#markdown-worksheet)
