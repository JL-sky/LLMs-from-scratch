import torch.nn as nn
import torch
class MutilHeadAttention(nn.Module):
    def __init__(self, d_in,d_out,context_length,dropout,head_nums,qkv_bias=False) -> None:
        super().__init__()
        assert (d_out%head_nums==0),"d_out 必须被head_nums整除"

        self.W_query=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value=nn.Linear(d_in,d_out,bias=qkv_bias)
        # 线性投影层，组合头的输出
        self.out_proj=nn.Linear(d_out,d_out)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )
        # 计算头的数量
        self.d_out=d_out
        self.head_nums=head_nums
        self.head_dim=d_out//head_nums
    def forward(self,x):
        b,num_tokens,d_in=x.shape

        # shape:b,num_tokens,d_out
        queries=self.W_query(x)
        keys=self.W_key(x)
        values=self.W_value(x)

        # 拆分d_out维度，将其转换为多个头
        queries=queries.view(b,num_tokens,self.head_nums,self.head_dim)
        keys=keys.view(b,num_tokens,self.head_nums,self.head_dim)
        values=values.view(b,num_tokens,self.head_nums,self.head_dim)
        # 转换维度，方便计算注意力分数
        # shape:(b,head_nums,num_tokens,head_dim)
        queries=queries.transpose(1,2)
        keys=keys.transpose(1,2)
        values=values.transpose(1,2)
        # 计算每个头的注意力分数
        atten_scores=queries@keys.transpose(2,3)
        # 计算每个头的掩码注意力分数，当词元数量小于context_length时进行截断
        atten_scores.masked_fill_(self.mask.bool()[:num_tokens,:num_tokens],-torch.inf)
        # 计算每个头的掩码注意力权重
        atten_weights=torch.softmax(atten_scores/keys.shape[-1]**0.5,dim=-1)
        # 随机dropout
        atten_weights=self.dropout(atten_weights)
        # 计算每个头的上下文向量
        context_vec=atten_weights@values
        # 还原矩阵形状
        context_vec=context_vec.transpose(1,2)
        # view() 只能作用于【内存连续】的张量,transpose() / permute() 交换维度后，张量会变成【内存不连续】
        context_vec=context_vec.contiguous().view(b,num_tokens,self.d_out)
        # 重新组合多头
        context_vec=self.out_proj(context_vec)
        return context_vec

if __name__=="__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
    )

    batch=torch.stack((inputs,inputs),dim=0)
    torch.manual_seed(123)
    batch_size,context_length,d_in=batch.shape
    d_out=2
    mha=MutilHeadAttention(d_in=d_in,d_out=d_out,context_length=context_length,dropout=0.0,head_nums=2)
    context_vecs=mha(batch)
    print(context_vecs)
    print(context_vecs.shape)
