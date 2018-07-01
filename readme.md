# 项目说明以及运行帮助

## 项目说明

本项目作为个人 Udacity Machine Learning Nano Degree 的毕业项目。项目问题取自 Kaggle 竞赛 [rossmann-store-sales](https://www.kaggle.com/c/rossmann-store-sales)。

本仓库包含项目实施的所有代码、文件。依据项目的实施顺序对仓库中的主要文件进行以下说明：

- /proposal 文件夹：项目开题报告

- /data 文件夹: kaggle提供的项目原始数据
- /submission 文件夹:  符合kaggle要求的submission文件
- /xgboost-python 文件夹：xgboost-gpu的安装文件

- First Glance.ipynb ：在着手处理数据和算法实施之前对数据集整体进行初步观察，对应报告的第一章“定义”
- Playground.ipynb ：对数据进行了非系统性的探索和观察，对应报告第二章“分析”
- Feature Engineering.ipynb : 对数据集进行了较为完善的特征工程预处理,对应报告3.1"数据预处理"
  - featuretank.py和value_filling.py分别为特征处理和特征填充过程中需要的代码,为了使用方便,将这两部分代码整理为脚本以供引用
- Predict.ipynb : 对数据使用多种算法进行了实施和分析,并得到了预期的研究成果,对应报告3.2至5.1的部分。
  - gcv-56core.py：由于个人电脑计算能力有限，随机森林的网格搜索部分单独放在云计算平台上运行，运行代码保存在该脚本中。

## 运行帮助

本项目的运行环境采用conda搭建，如果引用库中存在依赖关系，conda将自行安装依赖，建议同样使用conda环境进行结果复现。项目中的python环境以及需要的三方库均无特指版本，采用conda源中的最新版本即可

- python=3.6
- scikit-learn
- numpy
- matplotlib
- seaborn
- pandas
- jupyter lab

本项目中使用的xgboost-gpu需要单独安装，准备<https://github.com/dmlc/xgboost>的python包以及来自<http://www.picnet.com.au/blogs/guido/2016/09/22/xgboost-windows-x64-binaries-for-download/>的最新dll文件即可完成安装。本仓库的 /xgboost-python文件夹中已经包含上述文件，只需在 /xgboost-python路径下运行安装命令即可安装。

```shell
python setup.py install
```

项目的复现依照项目说明部分的文件顺序依次运行即可，需要说明的是，受制于ipython kernel，windows和linux平台上的并行计算（pools的应用以及基于pools的gridsearchcv等等）有一定的区别，需要进行简单的修改。