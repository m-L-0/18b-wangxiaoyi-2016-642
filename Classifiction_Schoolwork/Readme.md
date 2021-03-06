# Classifiction Schoolwork——使用支持向量机分类（svm）对九类别光谱进行分类
##### 河北师范大学软件学院  2016级 机器学学习方向  分类学习课程设计
#####              王晓意  2016011642
 
 #  一、设计思路


 1. 运用pca将样本降维，求出支持样本的排序数组

 2. 7：3分割数据集合，并建立九种一对九类别的svc

 3. 求得每种类别的预测结果矩阵，对每种类别的模型进行求正确率

 4. 调整参数使得每种模型的正确率尽可能高，判空率尽可能低

 5. 若干次学习后得出最最终预测结果


# 二、测试样本的正确率测试
运用测试集合求取相应的正确率进行调参，使用训练集样本

```python
#建立9*测试集数量的矩阵，行为九次svc的类别计数器
predict_lebal=np.zeros([num_test])
for i in range(9):
	lebal=i+1
```

```python
#对于第i类的样本标签one-hot化
y_train_hot=change(y_train,lebal,num_train)
y_test_hot=change(y_test,lebal,num_test)
```

```python
#建立svc模型,并且对当前类别i进行模型学习
svc=SVC(kernel='rbf',gamma=0.125,C=15)
svc.fit(x_train,y_train_hot)
predict=svc.predict(x_test)
get_acc(predict,y_test_hot,lebal)
get_predict(predict,predict_lebal,lebal)
```

```python
#输出最终的预测结果
print(predict_lebal[1:40:1])
print(get_acc(predict_lebal,y_test,10))
```

##### 程序运行结果
	  第 1 类ova样本分类，准确率为： 0.9066410009624639 %
      第 2 类ova样本分类，准确率为： 0.9384023099133783 %
      第 3 类ova样本分类，准确率为： 0.9932627526467758 %
      第 4 类ova样本分类，准确率为： 0.9947064485081809 %
      第 5 类ova样本分类，准确率为： 0.9985563041385948 %
      第 6 类ova样本分类，准确率为： 0.9456207892204043 %
      第 7 类ova样本分类，准确率为： 0.8681424446583254 %
      第 8 类ova样本分类，准确率为： 0.9504331087584216 %
      第 9 类ova样本分类，准确率为： 0.9947064485081809 %
      [7. 7. 6. 8. 7. 2. 8. 8. 9. 7. 5. 1. 7. 9. 9. 7. 7. 7. 0. 4. 9. 7. 4. 6.
       4. 3. 1. 4. 1. 9. 7. 0. 6. 1. 1. 7. 0. 1. 3.]
      第 10 类ova样本分类，准确率为： 0.7617901828681425 %
      None




# 三、算法性能分析

##### 1.现阶段参数设置：
*       pca降维，维度数为6
*       svc 核函数使用为'rbf'，gamma=0.125， c=30

##### 2.分析影响准确度的因素：
1.      降维，维度数在6-40 之间有小范围的浮动，本次由于使用svc的原因，尽可能的选择了较少的维度数
2.		一对多算法的本身限制了正确率，当为了精度将误差值C下调时，样本的空判（即九种分类器都没有对其做出具体的分类）升高，影响总体精度。
	
