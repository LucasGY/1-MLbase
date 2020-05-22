今天的主题是：**XGBoost的常用参数含义及一些常见的调参场景**

**相关代码已经上传至Github：https://github.com/LucasGY/1-MLbase**

目录如下
[TOC]

## 一、 XGBoost参数
XGB的参数类型又可以被分为：XGBoost框架参数（General parameters）、XGBoost 弱学习器参数（Booster parameters ）、命令行版本的参数（Command line parameters）以及其他参数

### 1.1 XGBoost框架参数（General parameters）
**1. booster [default= gbtree ]**
>booster决定了XGBoost使用的弱学习器类型，可以是默认的gbtree, 也就是CART决策树，还可以是线性弱学习器gblinear以及DART。一般来说，我们使用gbtree就可以了，不需要调参。
Which booster to use. Can be gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions.

**2. n_estimators=100**
>n_estimators则是非常重要的要调的参数，它关系到我们XGBoost模型的复杂度，因为它代表了我们决策树弱学习器的个数。这个参数对应sklearn GBDT的n_estimators。n_estimators太小，容易欠拟合，n_estimators太大，模型会过于复杂，一般需要调参选择一个适中的数值。

**3. objective [回归default=reg:squarederror] [二分类default=binary:logistic][多分类default=multi:softmax]**
>objective代表了我们要解决的问题是分类还是回归，或其他问题，以及对应的损失函数。具体可以取的值很多，一般我们只关心在分类和回归的时候使用的参数。
在回归问题objective一般使用reg:squarederror ，即MSE均方误差。二分类问题一般使用binary:logistic, 多分类问题一般使用multi:softmax。
更多要参考：https://xgboost.readthedocs.io/en/latest/parameter.html

**4. verbosity [default=1]**
>Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug). 

**5. n_jobs=0（改成-1 加速训练）**
> 控制算法的并发线程数

### 1.2 XGBoost 弱学习器参数（Booster parameters ）
**1. learning_rate【default=0.3】**
> learning_rate控制每个弱学习器的权重缩减系数，和sklearn GBDT的learning_rate类似，较小的learning_rate意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。所以这两个参数n_estimators和learning_rate要一起调参才有效果。当然也可以先固定一个learning_rate ，然后调完n_estimators，再调完其他所有参数后，最后再来调learning_rate和n_estimators。

**2. max_depth [default=6]**
> 控制树结构的深度，
数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，需要限制这个最大深度，
具体的取值一般要网格搜索调参。这个参数对应sklearn GBDT的max_depth。要注意，在训练深树时，XGBoost会大量消耗内存。

**3. gamma [default=0, alias: min_split_loss]**
> XGBoost的决策树分裂所带来的损失减小阈值。也就是我们在尝试树结构分裂时，会尝试最大数下式：
![20200522104027](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200522104027.png)

**4. min_child_weight [default=1]**
> 最小的子节点权重阈值，如果某个树节点的权重小于这个阈值，则不会再分裂子树，即这个树节点就是叶子节点。这里树节点的权重使用的是该节点所有样本的二阶导数的和。
![20200522104207](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200522104207.png)
min_child_weight越大，算法就越保守，控制过拟合越严格
对此参数的详细解释：https://stats.stackexchange.com/questions/317073/explanation-of-min-child-weight-in-xgboost-algorithm

**5. subsample [default=1]**
> 子采样参数，这个也是不放回抽样，和sklearn GBDT的subsample作用一样。
也就是说随机森林的每棵树的训练有可能会碰到训练集中两个样本重合的状况，因为是放回抽样；像GBDT\XGBoost训练每棵树的训练样本是不会重合的，只是subsample会控制整个训练集被选中的概率。
选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。初期可以取值1，如果发现过拟合后可以网格搜索调参找一个相对小一些的值。
Subsampling will occur once in every boosting iteration.

**6. colsample_bytree, colsample_bylevel, colsample_bynode [default=1]**
> 这三个参数都是用于特征采样的，默认都是不做采样，即使用所有的特征建立决策树。
colsample_bytree控制整棵树的特征采样比例，colsample_bylevel控制某一层（depth）的特征采样比例，而colsample_bynode(split)控制某一个树节点的特征采样比例。比如我们一共64个特征，则假设colsample_bytree，colsample_bylevel和colsample_bynode都是0.5，则某一个树节点分裂时会随机采样8个特征来尝试分裂子树。

**7. reg_alpha[default=0]/reg_lambda[default=1]**
> 这2个是XGBoost的正则化参数。reg_alpha是L1正则化系数，reg_lambda是L2正则化系数，在原理篇里我们讨论了XGBoost的正则化损失项部分：
![20200522104402](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200522104402.png)

## 二、常用的调参场景
### 2.1 控制过拟合
当您观察到较高的训练准确度但较低的测试准确度时，很可能遇到了过拟合问题。
通常，您可以通过两种方法控制XGBoost中的过拟合：
- 第一种方法是直接控制弱学习器的复杂度。
  - 这包括max_depth，min_child_weight和gamma。
- 第二种方法是增加随机性，以使训练对噪声具有鲁棒性。
  - 这包括subsample和colsample_bytree。
  - 您还可以减小learning rate。（需要与n_estimators配合调参）。

### 2.2 更快的训练表现
* 有一个名为的参数`tree_method`，请将其设置为`hist`或`gpu_hist`以加快计算速度。

### 2.3 处理不平衡的数据集
对于广告点击日志等常见情况，数据集非常不平衡。这可能会影响XGBoost模型的训练，有两种方法可以对其进行改进。
- 如果您只关心预测的整体效果指标（AUC）
  - 平衡正负负重 scale_pos_weight。scale_pos_weight用于类别不平衡的时候，负例和正例的比例。类似于sklearn中的class_weight。
  - 使用AUC进行评估
- 如果您关心预测正确的可能性
  - 在这种情况下，您无法重新平衡数据集,将参数设置max_delta_step为有限数（例如1）以帮助收敛

## 三、代码实战
### 3.1 准备玩具数据集并划分数据
数据集地址：

https://github.com/microsoft/LightGBM/tree/master/examples/regression

```python
path =  r'C:\Users\29259\Desktop'
df_train=pd.read_csv(path+r"\regression.train",header=None,sep='\t')
df_test=pd.read_csv(path+r"\regression.test",header=None,sep='\t')
y_train = df_train[0].values
y_test = df_test[0].values
X_train = df_train.drop(0, axis=1).values
X_test = df_test.drop(0, axis=1).values
```

### 3.2 模型超参数调优训练
**注意：**
1. 这里我用的是sklearn网格搜索，只不过**封装了自己加的东西**，例如可视化调参结果。
2. 这里使用的XGB，是sklearn接口的，方便。
3. 只针对某一个参数进行调优，方便看出欠/过拟合的现象

```python
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
refit = 'f1_micro'
scoring = {'f1_micro': 'f1_micro', 'Accuracy':  make_scorer(accuracy_score)}

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0) # n_splits=1才是留出法
cv = 3

xgb_param_grid = {'max_depth': range(1, 50, 1)} # hypara need to tuning


model_and_param = [ 
                    {"name":'xgb',
                     "estimator":xgb.XGBClassifier(random_state = 0,n_estimators=5),
                     "feature_sel":None,
                     "preprocessing":None,
                     "param_grid": xgb_param_grid,
                    },
                  ]

grid_search = GridSearch(param = model_and_param,
                         cv = cv,
                         scoring=scoring,
                         refit=refit,
                         plot_search = True)
all_gs_results = grid_search.fit(X_train,y_train)   
```
针对两个不同的评分方式绘制出train/val的得分曲线：
![20200522133731](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200522133731.png)
可以看出：
* 在`max_depth`比较小的时候，模型处于欠拟合train/val的分数都很低；
* 当`max_depth=6`时，验证集的分数最高；
* 随着`max_depth`的继续增加，模型开始产生过拟合现象，这个现象的明显标志就是train score很高，但val score没有之前高了。

这里展示的结果只针对max_depth, 其他的参数如果理解其含义的话也是类似的。详情可见GitHub上的Jupyter Notebook：

https://github.com/LucasGY/1-MLbase

---
![20200522132557](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200522132557.png)

---











