
# 关系提取项目
## 主要内容
* 项目功能
* 项目结构
* 使用方法
* 参考资料
* 补充说明

## 项目功能
利用具有标签的关系提取样本训练关系提取模型

## 项目结构
* /data 存放转换后的训练数据，最终训练用到的是/Healthtrain_word.npy,/Healthtrain_pos1.npy,/data/Healthtrain_pos2.npy,/Healthtrain_y.npy
* /model 存放训练好的模型
* /origin_data 存放原始的训练数据，
** 其中/RETrainData.txt 为我最开始从医学文本里面提取的样本。对应样本关系放在/relation2id.txt文件
** /RETrainDataFinal 和 /RETrainDatanew为尝试提取因果关系的样本集，它们的样本关系具有的种类详见/relationRETrainDataFinal 和/relationRETrainDatanew
* /initial.py 转换原始的训练数据到需要的格式
* /changesample.py 转换原始训练集的关系种类，将实体反向，扩充样本集
* /network.py 网络结构定义 ，参数设置
* /test_GRU.py 测试文件
* /train_GRU.py 训练文件

## 使用方法
### Requrements

* Python (>=3.5)

* TensorFlow (>=r1.0)

* scikit-learn (>=0.18)

### Training:

1. 把原始训练数据放到 origin_data/ , 包括关系标签文件 (relation2id.txt), 训练数据(RETrainData.txt), 测试数据(RETestData.txt) 和 中文词向量(vec.txt).

```
现在包含3种关系：
因果，果因，不明确
```

2. 将原始训练数据转化为 npy 文件, 保存在 data/
```
#python3 initial.py
```

3. Training, 模型会被保存在 ／model／
```
#python3 train_GRU.py
```


### Test，记得在test_GRU.py 里面修改用到的模型地址

```
#python3 test_GRU.py
```

## 参考资料
* https://github.com/thunlp/TensorFlow-NRE
*  [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://anthology.aclweb.org/P16-2034) 
*  [Neural Relation Extraction with Selective Attention over Instances](http://aclweb.org/anthology/P16-1200)
* 网络模型
![](http://www.crownpku.com/images/201708/1.jpg)

![](http://www.crownpku.com/images/201708/2.jpg)

## 补充说明
* 现在的样本集还没有做成具有 因果，果因，不明确 三种关系的样本集，可以在/changesample.py 基础上进行修改，
* 目前用/RETrainDataFinal 和 /RETrainDatanew 训练的模型还不能正确的区分出同一句话里面的<症状，疾病>和<疾病，疾病>关系
* 训练样本例：感冒	畏寒	关系  有感冒的病人表现为畏寒、发热、口淡无味、头痛、头胀、腹痛、腹泻等症状。
