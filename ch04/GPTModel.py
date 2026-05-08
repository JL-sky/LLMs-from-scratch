import sys
sys.path.append("..")  
from ch03.mutiAttention import MutilHeadAttention
import torch
import torch.nn as nn
from torchinfo import summary
import tiktoken

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))  
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
# Transformer decoder
class TransformerBlock(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.att=MutilHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            head_nums=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
            )
        self.ff=FeedForward(cfg)
        self.norm1=LayerNorm(cfg["emb_dim"])
        self.norm2=LayerNorm(cfg["emb_dim"])
        self.drop_shortcut=nn.Dropout(cfg["drop_rate"])

    def forward(self,x):
        # 原始残差
        shorcut=x
        # 第一层归一化
        x=self.norm1(x)
        # 多头注意力模块
        x=self.att(x)
        # dropout
        x=self.drop_shortcut(x)
        # 添加原始残差
        x=x+shorcut

        shorcut=x
        x=self.norm2(x)
        x=self.ff(x)
        x=self.drop_shortcut(x)
        x=x+shorcut

        return x
# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
        )
    
    def forward(self,x):
        return self.layers(x)

class GPTModel(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        # 嵌入所有词
        self.tok_emb=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        # 添加位置嵌入
        self.pos_emb=nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb=nn.Dropout(cfg["drop_rate"])

        self.trf_blocks=nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm=LayerNorm(cfg["emb_dim"])

        self.out_head=nn.Linear(
            cfg["emb_dim"],cfg["vocab_size"],bias=False
        )

    def forward(self,in_idx):
        batch_size,seq_len=in_idx.shape
        # shape:batch_size,seq_len(context_length),emb_dim
        tok_embeds=self.tok_emb(in_idx)
        # 占位符向量torch.arange(context_length)包含一个从0开始递增，直至最大输入长度减1的数值序列
        # 因此这里其实是把位置编号 [0, 1, 2, ..., seq_len-1]转换成位置向量 [seq_len, emb_dim]
        pos_embeds=self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x=tok_embeds+pos_embeds
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        """
        output shape:[bact_size,seq_len,vocab_size]
        第 1 维：batch（第几句话）
        第 2 维：seq_len（第几个词）
        第 3 维：vocab_size（词典里每个词的概率）
        """
        logits=self.out_head(x)
        return logits

# 给一段文本,让模型一个字一个字自动续写,直到生成 max_new_tokens 个字
# idx shape:[batch_size, seq_len]
def generate_text_simple(model,idx,max_new_tokens,context_length):
    for _ in range(max_new_tokens):
        # 使用滑动窗口，每次只把最近的 context_length 个词喂给模型
        idx_cond=idx[:,-context_length:]
        with torch.no_grad():
            # logits shape: [batch_size, seq_len, vocab_size]
            logits=model(idx_cond)
        # 模型会输出每一个位置的预测,但我们只需要最后一个位置的结果,因此shape变成[batch_size, vocab_size]
        logits=logits[:,-1,:]
        # 把输出值变成 0~1 的概率分布
        probas=torch.softmax(logits,dim=-1)
        # 选出概率最高的词
        idx_next=torch.argmax(probas,dim=-1,keepdim=True)
        # 将计算的下一个字符的索引添加到索引数组中，此时idx的shape为(batch,seq_len+1)
        idx=torch.cat((idx,idx_next),dim=-1)
    return idx

def TransformerTest(cfg):
    block=TransformerBlock(cfg)

    # torch.manual_seed(123)
    # x=torch.rand(2,4,768)
    # output=block(x)
    # print(x.shape)
    # print(output.shape)
    summary(block,input_size=(2, 4, 768))

def GPTModelTest(cfg,tokenizer):
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch.shape)

    torch.manual_seed(123)
    model=GPTModel(cfg)
    summary(model,input_size=(2,4), dtypes=[torch.int64])

def TextGenTest(cfg,tokenizer):
    start_context="Hello,I am"
    encoded=tokenizer.encode(start_context)
    print("encode",encoded)
    # unsqueeze将encoded从[4]转换为[1,4]，为其增加一个batch维度,来适应模型输入shape
    encoded_tensor=torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor:",encoded_tensor.shape)

    model=GPTModel(cfg)
    model.eval()
    out=generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_length=cfg["context_length"]
    )
    print("output:",out)
    print("output length:",len(out[0]))
    # squeeze(0)，去掉批次维度
    # tolist()，将PyTorch 张量变成普通 Python 列表
    decoded_text=tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)

if __name__ =="__main__":
    """
    batch_size = 2        → 一次看 2 句话
    context_length = 4    → 每句话最多 4 个词
    vocab_size = 50257    → 词典有 50257 个词
    emb_dim = 768         → 每个词用 768 维向量表示
    """

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # 总词汇表大小，即一共有多少个单词
        "context_length": 1024, # 上下文窗口长度，即将vocab_size进行切分后，每组的大小
        "emb_dim": 768,         # 词元 embeding 维度，将单个词嵌入后的维度大小
        "n_heads": 12,          # 注意力头数量
        "n_layers": 12,         # 网络层数
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-Key-Value bias
    }

    print("TransformerTest>>>>>>\n")
    TransformerTest(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")
    print("GPTModelTest>>>>>>\n")
    GPTModelTest(GPT_CONFIG_124M,tokenizer)
    print("TextGenTest>>>>>>\n")
    TextGenTest(GPT_CONFIG_124M,tokenizer)