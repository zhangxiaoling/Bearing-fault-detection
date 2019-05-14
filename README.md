Bearing-fault-detection （轴承故障检测）
========================
dc竞赛轴承故障检测训练赛：[比赛主页](https://www.dcjingsai.com/common/cmpt/%E8%BD%B4%E6%89%BF%E6%95%85%E9%9A%9C%E6%A3%80%E6%B5%8B%E8%AE%AD%E7%BB%83%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)

背景
----
轴承是在机械设备中具有广泛应用的关键部件之一。由于过载，疲劳，磨损，腐蚀等原因，轴承在机器操作过程中容易损坏。事实上，超过50％的旋转机器故障与轴承故障有关。实际上，滚动轴承故障可能导致设备剧烈摇晃，设备停机，停止生产，甚至造成人员伤亡。一般来说，早期的轴承弱故障是复杂的，难以检测。因此，轴承状态的监测和分析非常重要，它可以发现轴承的早期弱故障，防止故障造成损失。

任务
----
轴承有3种故障：外圈故障，内圈故障，滚珠故障，外加正常的工作状态。如表1所示，结合轴承的3种直径（直径1,直径2,直径3），轴承的工作状态有10类：

![](https://github.com/zhangxiaoling/Bearing-fault-detection/blob/master/data/1.png)

数据
----
1.train.csv，训练集数据，1到6000为按时间序列连续采样的振动信号数值，每行数据是一个样本，共792条数据，第一列id字段为样本编号，最后一列label字段为标签数据，即轴承的工作状态，用数字0到9表示。

2.test_data.csv，测试集数据，共528条数据，除无label字段外，其他字段同训练集。

总的来说，每行数据除去id和label后是轴承一段时间的振动信号数据，选手需要用这些振动信号去判定轴承的工作状态label。

数据分析
-------
通过readcsv.py（一开始不太会使用pandas读取和处理csv数据，因此先将dataframe格式的数据转换层list数据进行操作），showdata.py读取数据并画出前几条数据进行波形观察。波形如下，通过观察可以发现，不管是否存在故障以及存在何种故障，轴承的运行状况呈现周期。（这对接下来的数据增强提供了有效的帮助）

![](https://github.com/zhangxiaoling/Bearing-fault-detection/blob/master/data/data.png)

方案要素
--------
1、对时间序列进行数据增强。(cutdata.py,testdata_cutdata.py)
可用的训练集数量只有792条数据，数据量较小，使用CNN进行分类容易造成过拟合。因此可以将周期状态的6000维特征进行分割，切分成多段维度较小且相同的特征（经过实验可知，维度为3000，分割间隔为500）。对于每一条训练数据来说，可以数据增强为对应的七条训练数据。也就是说，在数据增强之后，原有训练集中的792条训练数据，增加到5544条训练数据，大大减少了CNN过拟合的可能性。

2、将CNN用于一维时间信号的特征提取与识别。(CNN.py)
将CNN用于1D的时间信号的特征提取与识别，具体的CNN框架如下：
![](https://github.com/zhangxiaoling/Bearing-fault-detection/blob/master/data/2.png)

3、采用投票法对分类结果进行故障分类。(result.py)
经过训练完毕的CNN分类器时会得到相应的七个标签，将这七个标签进行投票，得到概率最大的类别。

结果和排名
---------
![](https://github.com/zhangxiaoling/Bearing-fault-detection/blob/master/WeChat%20Image_20190417160034.png)

总结
------
这是一个基于深度学习的简单方案，传统的机器学习方法可见这位同学的![github](https://github.com/luanshiyinyang/DataMiningProject/tree/Bearing)

