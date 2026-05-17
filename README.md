# LLMs-from-scratch

## 环境准备

1.使用Jupyter虚拟环境

创建虚拟环境

```
conda create -n llm python=3.10
```

激活环境

```
conda activate llm
```

安装ipykernel

```
conda install ipykernel
```

将环境注册到 Jupyter

```
python -m ipykernel install --user --name llm --display-name "llm"
```

2.安装基本环境

选择自己合适的cuda版本，安装pytorch，官网如下：

> [Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/)

然后安装以下包

```
pip install tiktoken
pip install torchinfo
pip install matplotlib
pip install requests

```
