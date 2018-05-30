# 安装说明
- 需要环境JRE>=1.7，Python2.7
- Python相关包安装：pip install -r requirements.txt

# 运行说明
- python applilcation <IP地址> <端口号>
- 在浏览器端打开http://<IP地址>:<端口号>，假设设定的ip地址为114.212.190.232，端口号为33333,那么打开http://114.212.190.232:33333即可

# 文件说明
- data：存放神经网络解题模型及相关数据
- ner：实体识别程序及打包好的jar包，可直接使用jar包进行实体识别
- static：网页图片
- templates：网页html文件
- *.py：服务器提供的服务

# 数据、模型下载地址
神经网络相关模型以及数据可于百度云盘下载，存放于根目录下data文件夹
- 链接: https://pan.baidu.com/s/1Koogx37Kr8K64gJCgghNGA 
- 密码: 73gk

实体识别相关数据存放于ner/source_code/data文件夹中
- 链接: https://pan.baidu.com/s/151IM9Jd9tPJOziQZ7kUliA 
- 密码: b9x6
注：source_code提供给用户自定义NER程序，如需直接使用，直接在命令行输入java -jar ner.jar <inputFile> <outputFile>即可

