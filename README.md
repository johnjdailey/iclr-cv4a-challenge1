[ICLR Computer Vision for Agriculture Workshop](https://www.cv4gc.org/cv4a2020/)
[The CGIAR Wheat Rust Detection Challenge](https://zindi.africa/competitions/iclr-workshop-challenge-1-cgiar-computer-vision-for-crop-disease)

Authors: [Hanchung Lee](https://github.com/leehanchung), [Hursh Desai](https://github.com/hurshd0)

## Data
A total of 876 training images were provided by CGIAR.
`healthy_wheat`: 142
`leaf_rust`: 358
`stem_rust`: 376


## Evaluation Metric
The evaluation metric for this challenge is Log Loss.

Some images may contain both stem and leaf rust, there is always one type of rust that is more dominant than the other, i.e. you will not find images where both appear equally. The goal is to classify the image according to the type of wheat rust that appears most prominently in the image.

The values can be between 0 and 1, inclusive.

|ID       |leaf_rust   |stem_rust   |healthy_wheat   |
|---------|------------|------------|----------------|
|GVRCWM   | 0.63       | 0.98       | 0.21           |
|8NRRD6   | 0.76       | 0.11       | 0.56           |

##