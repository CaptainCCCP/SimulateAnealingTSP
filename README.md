# SimulateAnealingTSP
using SA(simulating aneanling) to solve TSP,with linear regression models to estimate temperature and total distance.
#使用模拟退火算法解决TSP问题
##简介：
这个程序主要使用python实现了模拟退火算法，将其用于解决TSP问题，并使用MindSpore框架实现了对退火温度和城市回路长度使用反向传播算法的线性回归预测。

##环境依赖
matplotlib==3.5.3
mindspore @ https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/cpu/x86_64/mindspore-1.9.0-cp38-cp38-win_amd64.whl
numpy==1.23.4
Pillow==9.3.0

##目录结构
├── Readme.md                 // help
├── defs.py                         // 函数定义
├── draw.py     //绘图
├── main.py     //主程序
└── predict.py                     //MindSpore相关函数主体

##使用说明
配置相应的环境，将所有.py文件放在同一目录下，运行main.py，如无报错将在终端看见退火算法结果打印
