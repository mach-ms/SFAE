# Steganalytic Feature Based Adversarial Embedding for Adaptive JPEG Steganography
## Iterative SFAE(ISFAE)和Oneshot SFAE(OSFAE)的源代码
### 环境
python 3.6  
Tensorflow 1.8  
操作系统：Windows 10  

### 数据
需要首先借助matlab准备相关数据  
jpeg系数使用Phi Sallee的jpeg toolbox读取，并保存为mat文件，变量名保存为coef  
jpeg量化表保存为mat文件，变量名保存为qtbl  
juniward代价同样需要事先计算，保存为mat文件，变量名保存为rho，由于+1 -1代价相同，因此只用保存一份即可  

### 运行
主程序是iter_sfae_q75_juni.py和ones_sfae_q75_juni.py，分别是Iterative SFAE和Oneshot SFAE的生成程序  
虽然名称指明了隐写代价函数juniward，但是也可以使用其他的jpeg域代价函数。同时，量化因子QF也可以  
在主程序中指定，但请和cover实际的质量因子匹配  

仅供研究使用。  
祝好:)

mach-ms


## Source code of SFAE(Steganalytic Feature Based Adversarial Embedding). Including Iterative SFAE(ISFAE) and Oneshot SFAE(OSFAE)

### ENV
Python 3.6  
Tensorflow 1.8  
OS: Windows 10  

### DATA
Before run the program, the JPEG data should be prepared:  
The JPEG data should be resolved in MATLAB with Phi Sallee's JPEG toolbox.  
JPEG DCT saved in .mat file, the variable name is 'coef'.  
JPEG QUANT table saved in .mat file, the variable name is 'qtbl'.  
JUNIWARD distortion should also be saved in .mat file before, the variable name is 'rho'.  

### RUN
Main file are iter_sfae_q75_juni.py(ISFAE) and ones_sfae_q75_juni.py(OSFAE).  
Note that you can also use other JPEG distortion function and JPEG QF.  

For research only.  
Have fun :)

mach-ms