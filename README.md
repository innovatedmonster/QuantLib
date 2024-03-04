<center> QuantLib </center>

## Introduction
QuantLib is an open source quantization toolbox based on PyTorch. 

## Supported methods
- [x] [LSQ (ICLR'2019)](configs/lsq) [LEARNED STEP SIZE QUANTIZATION](https://arxiv.org/abs/1902.08153)
- [x] [LSQ+ (CVPR'2020)](configs/lsq) [LSQ+: Improving low-bit quantization through learnable offsets and better initialization](https://arxiv.org/pdf/2004.09576.pdf)
- [x] [DAQ (ICCV'2021)](configs/daq) [Distance-aware Quantization](https://arxiv.org/abs/1902.08153)
- [x] [QIL (CVPR'2019)](configs/qil) [Quantization-interval Learning](https://arxiv.org/abs/1808.05779)

## Getting Started
### Dependencies
* Python == 3.7
* PyTorch == 1.8.2(我用的PyTorch==1.12.0+cu116)

[关于环境的坑](https://blog.csdn.net/Acemanindisguise/article/details/134851789?spm=1001.2014.3001.5501) 
> 各种坑 
>>关于DAQ:
>> + **DAQ部分代码写死了，只支持n bit**
>> + **DAQ就是在LSQ基础上的，也就是说DAQ本身就是改版rounding和LSQ的联合**
>> + DAQ没有使用self.beta时，acc91->acc88
>
>>关于QIL:
>> + QIL论文在实现时，必须**先量化后clamp**，如果先clamp后量化效果会很差，暂时不知道为什么
>> + 3b开始acc_qil骤降，daq和qil对wgt处理的区别是，qil是少了1个数的表示范围, daq是$[-2^{b-1}, 2^{b-1}-1]$而qil是$[-2^{b-1}, 2^{b-1}]$(都是对称量化，最终有符号)。

> 已经做的工作：
>> 最新版本是QEALSQ分支，LSQ部分修复了scale和其梯度形状不对应的bug, DAQ部分增加了关键注释和修复了resnet20_daq_W1A32.yml的路径bug
>
>> 实现了QIL, withoutScale(缩放因子scale，用来恢复量化反量化后conv输出的表达范围,[-1, 1]->[min, max])的情况下,使用cifar10数据集, w8a8量化， 经测试能达到acc88.51，87.86，88.62，87.76，88.38，88.35;
<br/>withScale(缩放因子scale，用来恢复量化反量化后conv输出的表达范围,[-1, 1]->[min, max])的情况下,使用cifar10数据集, w8a8量化， 经测试能达到acc91.42, 91.35,  91.32, 91.53, 90.79
>
>>实现了QILPlus,即QIL+DASR,withScale_cifar10_w8a8_epoch400_acc91.20, 91.01,  91.38,  90.75, 91.31, 91.40, 90.95

> 测试
>>关于cifar10_4bit量化：
>>>lsq，acc86.16 86.06
<br/>daq, acc91.27 91.21
>>
>>>qil，acc89.80, 88.28
<br/>qil_plus, acc89.85, 90.06

>>关于cifar10_3bit量化：
>>>daq，acc90.62, 90.41, 90.06, 90.65, 90.47, 90.61, 89.91, 90.34

>>关于bit_prune量化：
>>>init_8bit，acc84.63_bit5.07, acc84.67_bit5.07

### Installation
* Clone github repository.
```bash
$ git clone https://github.com/innovatedmonster/QuantLib.git
```
* Install dependencies
<br/>nvidia-smi命令确认cuda版本是11.6后创建环境
```bash
$ conda create -n quantlib python=3.7
```

```bash
$ pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 torchtext==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

```bash
$ pip install --upgrade setuptools==57.5.0
```

```bash
$ pip install apex
```

```bash
$ pip install -r requirements.txt
```


### Datasets
* Cifar-10
    * This can be automatically downloaded by learning our code, you can config the save path in the '*.yaml' file.
* ImageNet
    * This is available at [here](http://www.image-net.org) 

### Training & Evaluation
Cifar-10 dataset (ResNet-20 architecture) 

* First, download full-precision model into your folder(you can config the model path in your *.yaml file). **Link: [[weights](https://drive.google.com/file/d/1II9jtowxaGYde8_rYLs-qnPwzVcB3QYZ/view?usp=sharing)]**

```bash
# DAQ: Cifar-10 & ResNet-20 W1A1 model
$ python run.py --config configs/daq/resnet20_daq_W1A1.yml
# DAQ Cifar-10 & ResNet-20 W1A32 model
$ python run.py --config configs/daq/resnet20_daq_W1A32.yml
# LSQ: Cifar-10 & ResNet-20 W8A8 model
$ python run.py --config configs/lsq/resnet20_lsq_W8A8.yml
# LSQ+: Cifar-10 & ResNet-20 W8A8 model
$ python run.py --config configs/lsq_plus/resnet20_lsq_plus_W8A8.yml
```

## Results 
#### **Note**
* Weight quantization: Signed, Symmetric, Per-tensor. [Why use symmetric quantization.](https://www.qualcomm.com/media/documents/files/presentation-enabling-power-efficient-ai-through-quantization.pdf)
* Activation quantization: Unsigned, Asymmetric.
* Don't quantize the first and last layer. 

| methods | Weight | Activation | Accuracy | Models
| ------ | --------- | ------ | ------ | ------ |
| float | - | - | 91.4 | [download]() |
| LSQ | 8 | 8 | 91.9 | [download]() |
| LSQ+ | 8 | 8 | 92.1 | [download]() |
| DAQ | 1 | 1 | 85.8 | [download](https://drive.google.com/file/d/1zq8zZO_YnrLkMPybzZLJEBuSg66eFV4g/view) |
| DAQ | 1 | 32 | 91.2 | [download](https://drive.google.com/file/d/1SKHmms5kRLF_nLHf0qPbEO0JUOr34O5a/view?usp=sharing) |


## development notes
### bit_prune
>2024.3.1. **关于为什么bit会无限减小**
>>本来打算直接使用朴素量化，即plain scale的计算方式，但如果bit不可导，
bit也无法在loss函数中非惩罚项部分起到优化作用，即无法梯度更新，而仅仅在惩罚项中进行梯度更新(这导致什么问题？)
>
>>这将导致bit只会不断减小，而不收敛。
观察loss的权重衰减，设损失函数是$C=C_0+\frac{\lambda}{2n}\sum_ww^2$
则关于w的导数是$\frac{\partial C}{\partial w}=\frac{\partial C_0}{\partial w}+\frac{\lambda}{n}w$
而w的更新策略是:
$\begin{aligned}
&w\to w-\eta\frac{\partial C_{0}}{\partial w}-\frac{\eta\lambda}nw \\
&=\left(1-\frac{\eta\lambda}n\right)w-\eta\frac{\partial C_{0}}{\partial w}.
\end{aligned}$
 所以，这意味着，只在惩罚项中存在bit系数，bit会无限减小；而原损失函数部分确保了bit的更新有增大的可能。
 >
 >>**故应该令bit是连续可导，且使用plain scale而不是lsq更新scale**，其实现见models.quantization.bit_prune.bit_prune_bGradpScale.py
 >
 >>不对，bit的可导已经体现在插值量化中了。(但只是alpha_bit可导，bit不可导，也许这不够)

## Acknowledgement

QuantLib is an open source project. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.

## License

This project is released under the [MIT license](LICENSE).

---
### References
* https://github.com/ZouJiu1/LSQplus
* https://github.com/cvlab-yonsei/DAQ
* https://github.com/iimmortall/QuantLib