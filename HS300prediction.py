import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('/Users/yulipeng/Desktop/HS300.csv')

#统计原始数据涨跌变化
df=(data['pct_chg'].values>0)

count_00=0
count_01=0
count_10=0
count_11=0

for i in range(len(df)-1):
    if (df[i]==0 and df[i+1]==0):
        count_00=count_00+1
    elif (df[i]==0 and df[i+1]==1):
        count_01=count_01+1
    elif (df[i]==1 and df[i+1]==0):
        count_10=count_10+1
    else:
        count_11=count_11+1

print(count_00,count_01,count_10,count_11)

#调整数据，将这一期指标与下一期涨跌幅变化对齐
df1=pd.DataFrame(data['pct_chg'][1:len(data)].values)
df2=pd.DataFrame(data[['ATR','CCI','MACD','MTM','ROC','RSI']][0:-1].values)
df3=pd.concat([df1,df2],axis=1)
df3.columns=['pct_chg','ATR','CCI','MACD','MTM','ROC','RSI']

#绘制相关性图片
import seaborn as sns
g=sns.pairplot(df3, markers='d', size=2.5, plot_kws=
    {"s":40,
    "alpha":1.0,
    'lw':0.5,
    'edgecolor':'k'})
plt.show()

f, axes = plt.subplots(1, 1, figsize=(12, 12), sharex=True)
cols=df3.columns
cormat = np.corrcoef(df3[cols].values.T)
sns.set(font_scale=1.2)
heatmap = sns.heatmap(cormat,
                       cbar=True,
                       annot=True,
                       square=True,
                       fmt='.2f',
                       annot_kws={'size': 12},
                       yticklabels=cols,
                       xticklabels=cols)
plt.title('Correlation Matrix')
plt.show()

#处理数据，使其标准化
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

y = (df3['pct_chg'].values > 0)
X = df3.iloc[:,1:df3.shape[1]]

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=0)
    
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

#定义学习曲线绘制函数
def learning_curve_plot(clf,X,y,clfname):        
    train_sizes, train_scores, test_scores =\
                learning_curve(estimator=clf,
                               X=X,
                               y=y,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='Training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='Test accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.title(clfname)
    plt.xlabel('Training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.ylim([0.4, 1.1])
    plt.tight_layout()
    plt.show()
    return

from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.learning_curve import learning_curve


method = [
              ('logistic regression', LogisticRegression(random_state=1)),\
              ('svm.linear', svm.SVC(kernel='linear', gamma=0.2)),\
              ('svm.poly',svm.SVC(kernel='poly', gamma=0.2)),\
              ('svm.rbf',svm.SVC(kernel='rbf',gamma=0.2)),\
              ('svm.sigmoid',svm.SVC(kernel='sigmoid',gamma=0.2)),\

              ]
X_std = stdsc.transform(X)



for i in range(len(method)):
    pipe_cl = Pipeline([('scl', StandardScaler()),
                    method[i]])

    pipe_cl.fit(X_train, y_train)
    print('Method: %s (n=6)' % method[i][0])
    print('Training accuracy: %.3f\nTest  accuracy: %.3f' % (pipe_cl.score(X_train, y_train),pipe_cl.score(X_test, y_test)))

 
    learning_curve_plot(pipe_cl, X_train, y_train, method[i][0])
    
    y_pred = pipe_cl.predict(X_test)


#分析影响因子
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0)
forest.fit(X_train_std, y_train)
importances = forest.feature_importances_
feat_labels = df3.columns[1:df3.shape[1]]
indices = np.argsort(importances)[::-1]

#画出因子重要性
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

print(importances[indices])

#选取三因子进行研究
x=X.loc[:,['ATR','MTM']]
y=(df3['pct_chg'].values > 0)

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=0)


x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.transform(x_test)


#简化一些符号
from sklearn.tree import DecisionTreeClassifier

decision = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=5,
                              random_state=0)

from sklearn.ensemble import RandomForestClassifier

random = RandomForestClassifier(criterion='entropy',
                                n_estimators=10, 
                                random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

from sklearn.ensemble import ExtraTreesClassifier

extra =  ExtraTreesClassifier(criterion='entropy', 
                              max_depth=5,
                              random_state=0)

from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(base_estimator=decision,
                         n_estimators=10, 
                         learning_rate=0.5,
                         random_state=0)

from sklearn.ensemble import GradientBoostingClassifier

gradient = GradientBoostingClassifier(max_depth=5,
                              random_state=0)

from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(base_estimator=decision,
                        n_estimators=10, 
                        max_samples=10, 
                        bootstrap=True, 
                        bootstrap_features=False, 
                        n_jobs=1)

from sklearn.naive_bayes import GaussianNB

bayes = GaussianNB()

from sklearn.ensemble import VotingClassifier

vote = VotingClassifier(estimators=[('decision', decision), ('extra', extra)], voting='soft', weights=[1,1])

#定义分区域函数
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def pl(X,y,cl):
    # 设置颜色
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y_test))])

    # 画决策平面
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.05),
                           np.arange(x2_min, x2_max, 0.05))
    
    Z = cl.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(X[y == l, 0], X[y == l, 1], alpha=0.68,
                    c=c, label=l, marker=m)

    wrong_y = (y_test!=y_pred)
    plt.scatter(x=X[wrong_y==True,0],y=X[wrong_y==True,1],
                c='black',
                alpha=1,
                marker='v',
                s=66, label='Wrong prediction')

    
    plt.xlabel('ATR')
    plt.ylabel('MTM')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

#尝试不同方法
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

for clf, md in zip([decision, random, knn, extra, adaboost, gradient, bagging, bayes, vote],\
                   ['Decision Tree','Random Forest', 'KNN','Extra Trees','AdaBoost','Gradient Boosting','Bagging','Navie Bayes','Voting']):
   
        clf.fit(x_train_std, y_train)
    
        y_train_pred = clf.predict(x_train_std)
        y_test_pred = clf.predict(x_test_std)
    
        print('Method: %s ' % md)
        print('Training accuracy: %.3f\nTest  accuracy: %.3f' % (accuracy_score(y_train, y_train_pred),accuracy_score(y_test, y_test_pred)))

        learning_curve_plot(clf, X_train, y_train, md)
        
        y_pred=clf.predict(x_test_std)
        wrong_y = (y_test!=y_pred)

        pl(x_test_std,y_test,clf)
        
        #画confusion matrix
        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
     
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.show()

