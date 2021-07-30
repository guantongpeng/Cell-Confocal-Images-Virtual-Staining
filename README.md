### 虚拟染色

-------

本项目是基于飞桨框架的细胞虚拟染色技术实现，采用的模型是基于自注意力机制的3D Res-UNet。项目实现参考了：https://github.com/AllenCellModeling/pytorch_fnet/tree/release_1以及飞桨官方平台：https://www.paddlepaddle.org.cn/

--------

#### 数据集

数据集下载地址：http://downloads.allencell.org/publication-data/label-free-prediction/index.html
数据集名称与图像数据对应关系：
nucleoli ---- fibrillarin
nuclear envelope ---- lamin_b1 
microtubule images ---- alpha_tubulin
actin filament ---- beta_actin 
mitochondria ---- tom20 
cell membrane ---- membrane_caax_63x
endoplasmic reticulum ---- sec61_beta
dna ---- myosin_iib
(DIC) nuclear envelope ---- dic_lamin_b1
actomyosin bundles ---- myosin_iib
tight junctions ---- zo1
golgi ---- st6gal1
desmosome ---- desmoplakin

------

#### 模型训练

终端远程代码示例：

```shell
./scripts/train.sh  [dataset] [gpu_id] [model_name]
```

以fibrillarin为例，首先cd到项目文件夹下，终端运行`./scripts/train.sh fibrillarin -1 gvtnet`。

其中fibrillarin表示我们要训练细胞核仁预测模型，-1表示使用CPU进行训练，gvtnet表示我们将我们的模型命名为gvtnet。其中dic_lamin_b1` and `membrane_caax_63x例外，其训练命令如下：

```shell
./scripts/train_dic.sh [gpu_id] [model_name]
./scripts/train_membrane.sh [gpu_id] [model_name]
```

--------

#### 模型预测

终端远程代码示例：

```
./scripts/predict.sh  [dataset] [gpu_id] [model_name] [checkpoint_num]
```

以fibrillarin为例，首先cd到项目文件夹下，终端运行`./scripts/predict.sh fibrillarin -1  gvtnet 10000`，其中dic_lamin_b1` and `membrane_caax_63x例外，其预测命令如下：

```shell
./scripts/predict_dic.sh [gpu_id] [model_name] [checkpoint_num]
./scripts/predict_membrane.sh [gpu_id] [model_name] [checkpoint_num]
```



