import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor


'''
print("当前工作目录:", os.getcwd())  # 看看 Python 认为你在哪
print("脚本所在目录:", os.path.dirname(os.path.abspath(__file__)))  # 看看脚本实际在哪
'''

#1.====================读取数据====================

# 获取当前脚本所在的文件夹路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 读取 训练集（全部用来训练模型）
train_df = pd.read_csv(os.path.join(script_dir, 'train.csv'), )
# 读取 测试集（用来预测房价，没有SalePrice列）
test_df = pd.read_csv(os.path.join(script_dir, 'test.csv'), )
test_ids = test_df['Id']               #保存测试集ID
# 验证一下
print("成功读取文件！")

#数据打印函数
def data_print(df):
    print(f"数据形状: {df.shape}")
    print("\n=== 前五行数据 ===")
    print(df.head())
    print("\n=== 数据类型与缺失值总览 ===")
    df.info()  # 单独写，自动打印
    print("\n=== 数值列统计信息 ===")
    print(df.describe())
    print("\n=== 每列缺失值数量（从多到少） ===")
    print(df.isnull().sum().sort_values(ascending=False))

# data_print(train_df)
# data_print(test_df)

#2.==================数据清洗=====================

#2.1=================训练集数据清洗函数===============
def clean_train(df):
    #删除严重缺失的列
    missing_rate=df.isnull().sum()/len(df)

    drop_cols=missing_rate[missing_rate>0.8].index      #找出缺失率大于百分之五十的列

    df=df.drop(drop_cols,axis=1)        #axis=1删掉列，axis=0删掉行
    
    #填充剩余的空值
    #填充数字列的空缺
    num_cols=df.select_dtypes(include=['int64','float64']).columns       #.columns拿到数字列的列名
    for col in num_cols:
        df[col]=df[col].fillna(df[col].median())                         #用中位数填充

    #填充文字列空缺
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # print("\n检验填充结果:")
    # print(df.isnull().sum().sort_values(ascending=False))

    #删重复行
    df = df.drop_duplicates()

    #处理训练集中的异常值,保留合理房价
    Q1 = df['SalePrice'].quantile(0.25)     #百分之二十五位置的价格
    Q3 = df['SalePrice'].quantile(0.75)     #百分之七十五位置的价格
    IQR = Q3 - Q1
    df = df[(df['SalePrice'] >= Q1 - 1.5*IQR) & (df['SalePrice'] <= Q3 + 1.5*IQR)]         #保留房价在 [Q1-1.5*IQR ~ Q3+1.5*IQR] 范围内的房子

    print("\n========== 训练集清洗后最终检查 ==========")
    print(f"\n训练集最终数据形状: {df.shape}")

    return df,drop_cols

#2.2==================测试集数据清洗函数======================
#测试集的每一行都不能删除，且列数必须与训练集保持一致

def clean_test(df,drop_cols):
    # 只删除和训练集一样的列，保证列一致
    df = df.drop(drop_cols, axis=1)
    # 填充缺失值（不删行）
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\n========== 测试集清洗后最终检查 ==========")
    print(f"\n测试集最终数据形状: {df.shape}")
    
    return df

#2.3================执行数据清洗=================
train_df, drop_cols = clean_train(train_df)#训练集
test_df = clean_test(test_df, drop_cols)#测试集



#3.==================KNN算法模型预测=====================

'''
KNN算法原理:（KNN是一种有监督的机器学习方法）
KNN算法分为分类和回归算法，分类用于处理标签为非数字的数据，回归用于处理标签为数字的数据。
KNN预测的原理是计算预测样本值与测试集每个样本间的欧式距离之和，选出距离最小的k个样本，根据这些样本的标签来进行预测。
'''

#合并训练集（去掉房价）+ 测试集,为了后面统一文字转数字
all_data = pd.concat([train_df.drop('SalePrice', axis=1), test_df], axis=0)

#找出总数据里所有 文字类型的列
cat_cols = all_data.select_dtypes(include=['object']).columns

#将文字转换为数字，KNN只认数字，不认文字
for col in cat_cols:
    le = LabelEncoder()  # 创建文字转数字工具，LabelEncoder是sklearn自带的转换工具，将其命名为le，充当翻译官
    # 把文字转成数字（比如 地段A=0，地段B=1）
    all_data[col] = le.fit_transform(all_data[col].astype(str))

#拆分回训练集：总数据的前 N 行（N=训练集总行数）
train_processed = all_data.iloc[:len(train_df), :]
#拆分回测试集：总数据从 N 行开始的所有行
test_processed = all_data.iloc[len(train_df):, :]

# ======================准备特征======================
# 训练集特征：所有房屋信息（面积、房间数等）
X_train = train_processed
# 训练集标签：真实房价（模型要学习的目标）
y_train = train_df['SalePrice']
# 测试集特征：待预测的房屋信息
X_test = test_processed

# ====================== 数据标准化======================
scaler = StandardScaler()  # 创建标准化工具
X_train = scaler.fit_transform(X_train)  # 训练集
X_test = scaler.transform(X_test)        # 测试集

# ======================KNN模型训练 + 预测======================

#利用KNN回归模型
# 创建KNN模型，找 5 个最近的数字
knn = KNeighborsRegressor(n_neighbors=5)
# 训练模型
knn.fit(X_train, y_train)
# 预测：给测试集的房子算出房价
predictions = knn.predict(X_test)

# ======================生成提交文件 ======================
# 制作提交表格：ID + 预测房价
submission = pd.DataFrame({    #pd.DataFrame是pandas制作表格的函数
    'Id': test_ids,          # 测试集原始ID
    'SalePrice': predictions # 模型预测的房价
})
# 保存为CSV文件，index=False=不生成多余行号
save_path = os.path.join(script_dir, 'house_price_submission.csv')
submission.to_csv(save_path, index=False)