## 任务描述

[kaggle 房价预测竞赛](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)

请使用机器学习算法对房价进行预测

请将你的解答过程和结果传到 "solutions/你的专属文件夹" 下

./dataset/kaggle.ipynb 是一个参考的解答

## Kaggle 平台相关知识普及

后续需要在 kaggle 平台提交结果，因此需要预先进行配置

### 如何开始？

1. 获取 kaggle 命令
推荐：
```bash
conda create -n kaggle python=3.12
conda activate kaggle
pip install kaggle
```

2. 登录 [kaggle 官网](https://www.kaggle.com/) 注册账号

3. 在官网的设置页面点击 API Tokens (Recommended) 下的 **"Generate New Token"**，复制相应的 api token

4. 不要按照提示使用 export 命令，那是 Mac/Linux 系统的，Windows 用户点击设置->系统->高级系统设置->环境变量，在下面的 **系统变量** 里新建变量，变量名为"KAGGLE_API_TOKEN"，变量值为之前复制的 api token，然后一路点击确认

5. 重启终端，执行命令 "kaggle competitions list"，如果输出了很多比赛的网址说明可以使用了

### 相关命令

- 查看当前 kaggle 配置
```bash
kaggle config view
```

- 查看当前可以参加的竞赛
```bash
kaggle competitions list
```

- 下载比赛数据
```bash
kaggle competitions download -c 比赛名
```

- 提交比赛结果（需要先在相应比赛页面上报名）
```bash
kaggle competitions submit -c 比赛名 -f 文件名  -m "描述信息"
```

todo