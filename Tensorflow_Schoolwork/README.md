# Tensorflow Schoolwork——使用knn分类对鸢尾花数据集进行分类
##### 河北师范大学软件学院  2016级 机器学学习方向  tensorflow 课程实训作业
#####              王晓意  2016011642

# 一、设计思路

1. k近邻的方式对样本分类
2. 使用sklearn中split函数按照8：2的比例分割训练集和测试集

# 二、算法设计
```python
	建立预测结果集
    pred_list=[]
    acc=0
    accuracy=0
```
```python
 for i in range(n):
        #计算当前 样本 与 训练集 的距离
        near_dis=sess.run(distance,feed_dict={Xtr:X_train,Xte:X_test[i]})
        #排序后，按距离远近取前k个值
        knn=np.argsort(near_dis)[:k]
        #分三类别计数器，取数量最多的类别为样本类别
        label=[0,0,0]
        #根据k近邻内的样本标签对测试样本进行分类
        for j in knn:
            if(Y_train[j]==0):
                label[0]+=1
            elif(Y_train[j]==1):
                label[1]+=1
            else:
                label[2]+=1
        #取最大值
        final_label=np.argmax(label)    
        pred_list.append(final_label)
```
##### 程序运行结果
```python
预测运行结果为
Test 0 Predict_result: 0 True_class: 0
Test 1 Predict_result: 1 True_class: 1
Test 2 Predict_result: 1 True_class: 1
Test 3 Predict_result: 0 True_class: 0
Test 4 Predict_result: 2 True_class: 2
Test 5 Predict_result: 1 True_class: 1
Test 6 Predict_result: 2 True_class: 2
Test 7 Predict_result: 0 True_class: 0
Test 8 Predict_result: 0 True_class: 0
Test 9 Predict_result: 2 True_class: 2
Test 10 Predict_result: 1 True_class: 1
Test 11 Predict_result: 0 True_class: 0
Test 12 Predict_result: 2 True_class: 2
Test 13 Predict_result: 1 True_class: 1
Test 14 Predict_result: 1 True_class: 1
Test 15 Predict_result: 0 True_class: 0
Test 16 Predict_result: 1 True_class: 1
Test 17 Predict_result: 1 True_class: 1
Test 18 Predict_result: 0 True_class: 0
Test 19 Predict_result: 0 True_class: 0
Test 20 Predict_result: 1 True_class: 1
Test 21 Predict_result: 1 True_class: 1
Test 22 Predict_result: 1 True_class: 1
Test 23 Predict_result: 0 True_class: 0
Test 24 Predict_result: 2 True_class: 2
Test 25 Predict_result: 1 True_class: 1
Test 26 Predict_result: 0 True_class: 0
Test 27 Predict_result: 0 True_class: 0
Test 28 Predict_result: 1 True_class: 1
Test 29 Predict_result: 2 True_class: 2
finsh
预测正确率为：
accuracy: 1.0
```
#三、算法性能分析
##### 现阶段参数设置：k=1
##### 由于并未进行降维处理的因素、影响因素只有k本身
