import numpy as np
from scipy.sparse import isspmatrix, dok_matrix, csc_matrix
import sklearn.preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def normalize(matrix):
    #规范化矩阵
    #matrix 输入矩阵， return 规范化矩阵
    return sklearn.preprocessing.normalize(matrix,norm='l1',axis=0)

def inflate(matrix,power):
    #inflation操作,power为设定的r值
    return normalize(np.power(matrix,power))

def expand(matrix,power):
    #expand 操作,power为设定的e值
    return  np.linalg.matrix_power(matrix,power) 

def addloops(matrix,loop=1):
    '''
    添加节点回环，去除幂值的影响
    matrix：为输入的邻接矩阵
    loop：为回环大小，默认为1
    '''
    shape=matrix.shape
    new_matrix=matrix.copy()
    for i in range(shape[0]):
        new_matrix[i,i]=loop
    return new_matrix

def converged(matrix1,matrix2):
    #检查是否收敛
    return np.allclose(matrix1,matrix2)

def iterate(matrix,e,r):
    #进行expand操作
    #e为相应的e值
    matrix=expand(matrix,e)
    #inflate
    #r为r次幂方
    matrix=inflate(matrix,r)
    return matrix

def prune(matrix,min):
    #删除概率过小的边，min为最小概率
    #同时保护每列的最大值，用不删除最大值
    pruned=matrix.copy()
    pruned[pruned<min]=0
    num_cols = matrix.shape[1]
    row_indices = matrix.argmax(axis=0).reshape((num_cols,))
    col_indices = np.arange(num_cols)
    pruned[row_indices, col_indices] = matrix[row_indices, col_indices]
    return pruned

def run_mcl(matrix,e=2,r=2,min=0.01,max_turn=15):
    ''' matrix：输入邻接矩阵
        e：expand 对应参数e值，默认为2
        r:inflate 对应参数r值，默认为2
        max_turn:最大迭代次数，默认为15次'''
    
    for x in range(max_turn):
        matrix1=matrix.copy()
        matrix=iterate(matrix,e,r)        
        if converged(matrix1,matrix):
            break 
    print('共经历 %d 轮迭代后获得收敛矩阵'%(x))
    return matrix       

def get_clusters(matrix):
    if not isspmatrix(matrix):
        # cast to sparse so that we don't need to handle different 
        # matrix types
        matrix = csc_matrix(matrix)

    # get the attractors - non-zero elements of the matrix diagonal
    attractors = matrix.diagonal().nonzero()[0]

    # somewhere to put the clusters
    clusters = set()
    print(clusters)
    # the nodes in the same row as each attractor form a cluster
    for attractor in attractors:
        cluster = tuple(matrix.getrow(attractor).nonzero()[1].tolist())
        clusters.add(cluster)

    return sorted(list(clusters))


from sklearn.datasets import load_iris
# 导入iris 数据集
# 读取对应的特征集和标签数组

iris = load_iris()
data = iris.data
target = iris.target

pca=PCA(n_components=2)
pca.fit(data)
new_data=pca.fit_transform(data)
data_0=[]
data_1=[]
data_2=[]
num=len(data)
for i in range(num):
    if target[i]==0:
        data_0.append(new_data[i])
    if target[i]==1:
        data_1.append(new_data[i])
    if target[i]==2:
        data_2.append(new_data[i])
#转化为numpy矩阵
data_0_np=np.array(data_0)
data_1_np=np.array(data_1)
data_2_np=np.array(data_2)
plt.scatter(np.transpose(data_0_np)[0],np.transpose(data_0_np)[1])
plt.scatter(np.transpose(data_1_np)[0],np.transpose(data_1_np)[1])
plt.scatter(np.transpose(data_2_np)[0],np.transpose(data_2_np)[1])
plt.show()
distance_list=[]
#生成相应的150*150的邻接矩阵
for i in data:
    for j in data:
        #每个样本点与其他样本点的曼哈顿距离，用作度量相似度
        distance = np.abs(i[0]-j[0])+np.abs(i[1]-j[1])+np.abs(i[2]-j[2])+np.abs(i[3]-j[3])
        distance_list.append((distance))
# 转化为numpy形式
# 转化为150*150的邻接矩阵
distance_list=np.array(distance_list)
near_arr=distance_list.reshape(150,150)
#阈值初始设为2，后经过测试设为1.2
#修改图，祛除过远的链接度低的链接,转变成01矩阵
for i in range(150):
    for j in range(150):
        if near_arr[i][j]<=1.2 and near_arr[i][j]!=0:
            near_arr[i][j]=1
        else:
            near_arr[i][j]=0
#添加节点回环
new_near_arr=addloops(near_arr)
#运行mcl聚类算法
final_arr=run_mcl(new_near_arr)
#聚类结果

predict=get_clusters(final_arr)

#共分成5类别
#这里验证类别1，2只有两个样本差别所以合并为样本1
#x=np.allclose(predict[1],predict[2])

predict_0=[]
predict_1=[]
predict_2=[]
predict_3=[]
#建立一个将idex中的编号对应特征导入predict列表的函数
def charge(predict,n):
    num=len(predict[n])
    predict_n=[]
    for i in range(num):
        y=predict[n][i]
        predict_n.append(new_data[y])
    return predict_n
#建立对应类别的坐标数组，转变为np形式
predict_0=charge(predict,0)
predict_1=charge(predict,2)
predict_2=charge(predict,3)
predict_3=charge(predict,4)
predict_0_np=np.array(predict_0)
predict_1_np=np.array(predict_1)
predict_2_np=np.array(predict_2)
predict_3_np=np.array(predict_3)
plt.scatter(np.transpose(predict_0_np)[0],np.transpose(predict_0_np)[1])
plt.scatter(np.transpose(predict_1_np)[0],np.transpose(predict_1_np)[1])
plt.scatter(np.transpose(predict_2_np)[0],np.transpose(predict_2_np)[1])
plt.scatter(np.transpose(predict_3_np)[0],np.transpose(predict_3_np)[1])
#分类后的散点图
plt.show()
acc=0
total=len(predict_0)+len(predict_1)+len(predict_2)+len(predict_3)
#计算0类别正确率
for i in range(len(predict_0)):
    n=predict[0][i]
    if target[n]==0:
        acc+=1
#计算1类别正确率
for i in range(len(predict_1)):
    n=predict[1][i]
    if target[n]==1:
        acc+=1
#计算2类别正确率
for i in range(len(predict_2)):
    n=predict[3][i]
    if target[n]==2:
        acc+=1
#计算3类别正确率
for i in range(len(predict_3)):
    n=predict[4][i]
    if target[n]==3:
        acc+=1

#输出总体正确率结果
print('总体正确率结果为：%f'%(acc/total))