import pandas as pd
import numpy as np
import os
print("当前工作目录:", os.getcwd())  # 看看 Python 认为你在哪
print("脚本所在目录:", os.path.dirname(os.path.abspath(__file__)))  # 看看脚本实际在哪

#1.====================读取数据====================

# 获取当前脚本所在的文件夹路径
script_dir = os.path.dirname(os.path.abspath(__file__))
#拼接出 test.csv 的完整路径（无论脚本在哪，都能找到同文件夹下的 test.csv）
csv_path = os.path.join(script_dir, 'train.csv')
# 读取文件
df = pd.read_csv(csv_path)
# 验证一下
print("成功读取文件！")

print(f"数据形状: {df.shape}")
print("\n=== 前五行数据 ===")
print(df.head())
print("\n=== 数据类型与缺失值总览 ===")
df.info()  # 单独写，自动打印
print("\n=== 数值列统计信息 ===")
print(df.describe())
print("\n=== 每列缺失值数量（从多到少） ===")
print(df.isnull().sum().sort_values(ascending=False))

#2.==================数据清洗=====================

#2.1删除严重缺失的列
missing_rate=df.isnull().sum()/len(df)

drop_cols=missing_rate[missing_rate>0.8].index      #找出缺失率大于百分之五十的列

df=df.drop(drop_cols,axis=1)        #axis=1删掉列，axis=0删掉行

print(f"\n清洗后的数据形状{df.shape}")
print(f"\n被清理掉的列{list(drop_cols)}")        #显示清理后的数据

#2.2填充剩余的空值
#2.2.1填充数字列的空缺
num_cols=df.select_dtypes(include=['int64','float64']).columns       #.columns拿到数字列的列名
for col in num_cols:
    df[col]=df[col].fillna(df[col].median())                         #用中位数填充

#2.2.2填充文字列空缺
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n检验填充结果：")
print(df.isnull().sum().sort_values(ascending=False))

#2.3删重复行
df = df.drop_duplicates()
print(f"\n删除重复行后形状: {df.shape}")

#2.4处理异常值,保留合理房价
Q1 = df['SalePrice'].quantile(0.25)     #百分之二十五位置的价格
Q3 = df['SalePrice'].quantile(0.75)     #百分之七十五位置的价格
IQR = Q3 - Q1
df = df[(df['SalePrice'] >= Q1 - 1.5*IQR) & (df['SalePrice'] <= Q3 + 1.5*IQR)]         #保留房价在 [Q1-1.5*IQR ~ Q3+1.5*IQR] 范围内的房子

print("\n========== 清洗后最终检查 ==========")
print(f"\n最终数据形状: {df.shape}")

