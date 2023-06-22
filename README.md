# CVexp1



## 结果说明

### 结果位置

-   实验训练出的模型在runs文件夹中，最终的模型是best.pth，格式是torch的state_dict。
-   画图：TODO

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 项目介绍（如果有需要参考的童鞋可以主要看这块）

### 项目配置

-   如果得到本项目的方式是从github上获取的，那么首先需要安装对应的packages。注意，pytorch的版本要高于2.0。其余的包根据错误提示进行安装。主要使用包如下：
    -   numpy
    -   pandas
    -   tensorboard
    -   tqdm
    -   pytorch

-   项目数据集：https://www.kaggle.com/mloey1/ahcd1

-   其次，首次使用的时候，需要先将数据集中的包含图像的两个文件夹（倒数第二层，原名叫test和train）把它们copy到项目根目录的data路径下，如果没有，就创建一个。

-   训练之前，需要运行utils中的datapath.py，为数据创建csv的路径索引，程序中Dataloader类的构建会用到这个。懂得都懂，不懂的也不用懂，运行就行。

-   之后就可以进行训练了。训练、测试时可以加入超参数，这里如果有cuda和cudnn，可以考虑分别使用以下命令行运行（如果没有，就把-d和后面的东西都删了）：

```bash
python main.py -d cuda

python test.py -d cuda
```

根据经验，建议训练200个epoch及以上，我自己训练200轮，得到的结果已经超越了参考论文的水准，测试集上Acc可以达到96%。可以考虑使用命令行：

```bash
python main.py -d cuda --epochs 200

python test.py -d cuda
```

参考论文：https://arxiv.org/ftp/arxiv/papers/1706/1706.06720.pdf

作者：宸哥

guthub链接：https://github.com/beaut1fulb0y/CVexp1.git
