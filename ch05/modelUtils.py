import numpy as np
import torch
from ch04.GPTModel import generate_text_simple
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # 添加batch维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) 
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    # 移除batch维度
    flat = token_ids.squeeze(0) 
    return tokenizer.decode(flat.tolist())

"""
model:语言模型（如 GPT），输入 token IDs 输出 logits
idx:已生成的 token 序列，形状 (batch_size, num_tokens)
max_new_tokens:最多生成多少个新 token
context_size:模型能处理的最大上下文长度（滑动窗口）
temperature:温度参数，控制生成随机性（0=贪婪解码）
top_k:Top-K 采样，只从概率最高的 K 个 token 中选
eos_id:序列结束标记（End of Sequence），遇到则停止
"""
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # 每次循环生成 1 个新 token，直到达到 max_new_tokens 上限或遇到 eos_id
    for _ in range(max_new_tokens):
        # 只保留序列的最后 context_size 个 token 作为模型输入
        idx_cond = idx[:, -context_size:]
        # 模型预测
        with torch.no_grad():
            logits = model(idx_cond)
        # 语言模型是对每个输入位置都输出预测，但最后一个位置的下一个 token 预测才是我们想要的
        logits = logits[:, -1, :]

        # 执行top-k策略，避免选择概率极低的"离谱" token，提高生成质量
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # 温度缩放策略
        if temperature > 0.0:
            # # 温度缩放
            logits = logits / temperature
            # 数值稳定性处理
            """
            直接计算 e^x_j时，如果 logits 的值很大（比如几百），指数会爆炸性增长，导致：
                上溢（Overflow）：exp(1000) → inf
                下溢（Underflow）：分母变成 inf，结果变成 nan
            然而softmax 有一个重要性质：所有元素同时减去同一个常数，结果不变
            所以让logits 的每一行都减去该行的最大值，就能让所有指数运算都在可控范围内
            """
            logits = logits - logits.max(dim=-1, keepdim=True).values
            # # 转概率分布
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            #  # 按概率采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        # 正常计算概率值，贪婪解码，直接选概率最高的 token，完全确定性输出
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
        # 如果遇到序列结束词元，则提前停止生成
        # if idx_next == eos_id: 
        #     break
        if eos_id is not None and (idx_next == eos_id).all():
            break
        # 将新生成的 token 拼接到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

def load_hf_weights_into_custom_gpt(model, hf_weights, config, tie_lm_head=True):
    """
    加载 Hugging Face GPT-2 pytorch_model.bin 到你的自定义 GPTModel。

    适用场景：
        hf_weights = torch.load("gpt2/355M/pytorch_model.bin", map_location="cpu")
        load_hf_weights_into_custom_gpt(model, hf_weights, BASE_CONFIG)

    关键点：
        Hugging Face GPT-2 的 Conv1D.weight 是 [in_features, out_features]
        PyTorch nn.Linear.weight 是 [out_features, in_features]

        所以这些层必须强制转置：
            attn.c_attn.weight
            attn.c_proj.weight
            mlp.c_fc.weight
            mlp.c_proj.weight

        不能靠 shape 自动判断是否转置，因为 attn.c_proj.weight 是方阵。
    """

    emb_dim = config["emb_dim"]
    n_layers = config["n_layers"]

    # 兼容 checkpoint 外面包了一层的情况
    if isinstance(hf_weights, dict):
        for wrapper_key in ["state_dict", "model", "module"]:
            if wrapper_key in hf_weights and isinstance(hf_weights[wrapper_key], dict):
                hf_weights = hf_weights[wrapper_key]
                break

    def find_key(suffix):
        """
        支持这些 key 格式：
            h.0.attn.c_attn.weight
            transformer.h.0.attn.c_attn.weight
            model.transformer.h.0.attn.c_attn.weight
            module.transformer.h.0.attn.c_attn.weight
        """
        for k in hf_weights.keys():
            if k == suffix or k.endswith("." + suffix):
                return k
        raise KeyError(f"找不到以 {suffix} 结尾的权重 key")

    def get(suffix):
        return hf_weights[find_key(suffix)]

    def copy_param(dst, src, name):
        """
        把 src 拷贝到 dst，自动移动 device 和 dtype。
        """
        src = src.to(device=dst.device, dtype=dst.dtype)

        if dst.shape != src.shape:
            raise ValueError(
                f"{name} shape 不匹配: "
                f"target={tuple(dst.shape)}, source={tuple(src.shape)}"
            )

        dst.copy_(src)

    def copy_hf_conv1d_to_linear(linear, hf_weight, hf_bias, name):
        """
        Hugging Face GPT-2 Conv1D -> PyTorch nn.Linear

        HF Conv1D.weight:
            [in_features, out_features]

        nn.Linear.weight:
            [out_features, in_features]

        所以必须强制转置。
        """
        copy_param(
            linear.weight,
            hf_weight.T,
            f"{name}.weight"
        )

        if linear.bias is not None:
            copy_param(
                linear.bias,
                hf_bias,
                f"{name}.bias"
            )

    with torch.no_grad():
        # 1. Embedding
        copy_param(
            model.tok_emb.weight,
            get("wte.weight"),
            "tok_emb.weight"
        )

        copy_param(
            model.pos_emb.weight,
            get("wpe.weight"),
            "pos_emb.weight"
        )

        # 2. Final LayerNorm
        copy_param(
            model.final_norm.scale,
            get("ln_f.weight"),
            "final_norm.scale"
        )

        copy_param(
            model.final_norm.shift,
            get("ln_f.bias"),
            "final_norm.shift"
        )

        # 3. Output head
        # GPT-2 默认 lm_head 和 wte 共享权重
        if tie_lm_head:
            model.out_head.weight = model.tok_emb.weight
        else:
            try:
                copy_param(
                    model.out_head.weight,
                    get("lm_head.weight"),
                    "out_head.weight"
                )
            except KeyError:
                copy_param(
                    model.out_head.weight,
                    model.tok_emb.weight,
                    "out_head.weight"
                )

        # 4. Transformer blocks
        for i in range(n_layers):
            block = model.trf_blocks[i]
            bp = f"h.{i}."

            # LayerNorm 1
            copy_param(
                block.norm1.scale,
                get(f"{bp}ln_1.weight"),
                f"block.{i}.norm1.scale"
            )

            copy_param(
                block.norm1.shift,
                get(f"{bp}ln_1.bias"),
                f"block.{i}.norm1.shift"
            )

            # LayerNorm 2
            copy_param(
                block.norm2.scale,
                get(f"{bp}ln_2.weight"),
                f"block.{i}.norm2.scale"
            )

            copy_param(
                block.norm2.shift,
                get(f"{bp}ln_2.bias"),
                f"block.{i}.norm2.shift"
            )

            # Attention QKV
            qkv_w = get(f"{bp}attn.c_attn.weight")
            qkv_b = get(f"{bp}attn.c_attn.bias")

            expected_qkv_shape = (emb_dim, 3 * emb_dim)
            if tuple(qkv_w.shape) != expected_qkv_shape:
                raise ValueError(
                    f"block.{i}.attn.c_attn.weight shape 异常: "
                    f"got={tuple(qkv_w.shape)}, expected={expected_qkv_shape}"
                )

            expected_qkv_bias_shape = (3 * emb_dim,)
            if tuple(qkv_b.shape) != expected_qkv_bias_shape:
                raise ValueError(
                    f"block.{i}.attn.c_attn.bias shape 异常: "
                    f"got={tuple(qkv_b.shape)}, expected={expected_qkv_bias_shape}"
                )

            # HF c_attn.weight: [emb_dim, 3 * emb_dim]
            # 按列切成 Q/K/V，然后每个转置成 Linear 格式
            q_w, k_w, v_w = qkv_w.split(emb_dim, dim=1)
            q_b, k_b, v_b = qkv_b.split(emb_dim, dim=0)

            copy_param(
                block.att.W_query.weight,
                q_w.T,
                f"block.{i}.att.W_query.weight"
            )

            copy_param(
                block.att.W_key.weight,
                k_w.T,
                f"block.{i}.att.W_key.weight"
            )

            copy_param(
                block.att.W_value.weight,
                v_w.T,
                f"block.{i}.att.W_value.weight"
            )

            copy_param(
                block.att.W_query.bias,
                q_b,
                f"block.{i}.att.W_query.bias"
            )

            copy_param(
                block.att.W_key.bias,
                k_b,
                f"block.{i}.att.W_key.bias"
            )

            copy_param(
                block.att.W_value.bias,
                v_b,
                f"block.{i}.att.W_value.bias"
            )

            # Attention output projection
            # 注意：这里必须强制转置，不能用 shape 判断
            copy_hf_conv1d_to_linear(
                block.att.out_proj,
                get(f"{bp}attn.c_proj.weight"),
                get(f"{bp}attn.c_proj.bias"),
                name=f"block.{i}.att.out_proj"
            )

            # FFN c_fc: emb_dim -> 4 * emb_dim
            copy_hf_conv1d_to_linear(
                block.ff.layers[0],
                get(f"{bp}mlp.c_fc.weight"),
                get(f"{bp}mlp.c_fc.bias"),
                name=f"block.{i}.ff.layers.0"
            )

            # FFN c_proj: 4 * emb_dim -> emb_dim
            copy_hf_conv1d_to_linear(
                block.ff.layers[2],
                get(f"{bp}mlp.c_proj.weight"),
                get(f"{bp}mlp.c_proj.bias"),
                name=f"block.{i}.ff.layers.2"
            )

    print(f"HF 权重加载完成: {n_layers} 层, emb_dim={emb_dim}")
    return model

# 计算单批次损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# 计算所有批次的损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# 计算损失
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# 解码生成文本
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_length=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

# 模型训练
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # 初始化列表，记录损失
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # epoch迭代
    for epoch in range(num_epochs):
        model.train()  
        
        for input_batch, target_batch in train_loader:
            # 重置上一个批次迭代中的损失梯度
            optimizer.zero_grad() 
            # 计算损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 计算损失梯度
            loss.backward()
            # 使用损失梯度更新模型权重 
            optimizer.step() 
            tokens_seen += input_batch.numel()
            global_step += 1

            # 可选的评估步骤
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每轮之后打印一个文本样本
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  

    ax2 = ax1.twiny()  
    ax2.plot(tokens_seen, train_losses, alpha=0) 
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  
    plt.savefig("loss-plot.pdf")
    plt.show()
