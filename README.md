# DAT: Improving Adversarial Robustness via Generative Amplitude Mix-up in Frequency Domain (in NeurIPS 2024)
# Environment Settings in requirement.txt
python==3.9.1

pyTorch==1.8

Torchvision==0.8.0

Numpy==1.19.2

pillow==10.3.0

kornia==0.7.2

matplotlib==3.9.0

# Training
For training DAT on cifar-10 and cifar-100 on ResNet18: 
```
python train_cifar.py --arch ResNet18 --data CIFAR10
```
```
python train_cifar.py --arch ResNet18 --data CIFAR100
```
For training DAT on cifar-10 and cifar-100 on WideResNet34-10: 
```
python train_cifar.py --arch WideResNet34-10 --data CIFAR10
```
```
python train_cifar.py --arch WideResNet34-10 --data CIFAR100
```

For training DAT on Tiny-ImageNet: 
```
python train_tiny.py
```
# Evaluation

For evaluation of the trained model: 
```
python eval.py --arch {ResNet18/WideResNet34-10/WideResNet28-10} --data {CIFAR10/CIFAR100} --attack {AA/PGD}
```

# Citation

@inproceedings{DBLP:conf/neurips/dat24,
  author       = {Fengpeng Li and
                  Kemou Li and Haiwei Wu and
                  Jinyu Tian and
                  Jiantao Zhou},
  title        = {DAT: Improving Adversarial Robustness via Generative Amplitude Mix-up in Frequency Domain},
  booktitle    = {NeurIPS, December 9-15, 2024, Vancouver,
                  Canada},
  year         = {2024},
}
```
