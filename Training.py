######################################################################################################################################################
#                                               Name: Training_nanollama                                                                                   #
#                                               Comment Author: Devin Wu                                                                             #
#                                               Date: Jul, 10, 2023 10:25PM                                                                           #
#                                               Ref: http://pointborn.com/article/2022/2/18/1820.html                                                #
#                                                    https://blog.csdn.net/Mikeyboi/article/details/119522689                                        #
#                                                                                                                                                    #
######################################################################################################################################################


import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from Model_nanollama import GPTConfig, GPT

# 训练参数
out_dir = 'out'
eval_interval = 20
log_interval = 1
eval_iters = 50
eval_only = False
always_save_checkpoint = True
#init_from = 'scratch'

# 常规参数
dataset = 'Material' 
gradient_accumulation_steps = 5 * 8
batch_size = 6 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 512
# model参数
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False
# adamw参数
learning_rate = 6e-4
max_iters = 6000
weight_decay = 1e-1
beta1 = 0.9 #Adam 公式中的俩β
beta2 = 0.95
grad_clip = 1.0 # 梯度裁剪
# learning rate decay
decay_lr = True
warmup_iters = 200
lr_decay_iters = 2000
min_lr = 6e-5

# 系统参数
device = 'cpu'
dtype = 'bfloat16'
compile = False # Pytorch 2.0 新功能，需Cuda

torch.manual_seed(1337) #设定随机种子
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# Autocast, 只支持CUDA
# autocast 应该只封装网络的前向传播 (forward pass(es))，以及损失计算 (loss computation(s))。
# 反向传播不推荐在 autocast 区域内执行，反向传播的操作会自动以对应的前向传播的操作的数据类型运行。
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)
# 内存映射
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #随机取batch_size个数，范围是[0,len(data) - block_size]
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix]) #遍历上面ix的结果然后把窗口扩展到block_size大小作为input
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix]) #同上，遍历上面ix的结果然后把窗口扩展到block_size大小再shift right作为output/ground truth
    if device_type == 'cuda':
        # 如果你有cuda用，就把数据PIN到GPU
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0
best_val_loss = 1e9

# 看看下载的dataset里面有没有包含词汇表
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # 创建字典保存模型参数

# 从0开始训练
print("Initializing a new model from scratch")
# 定义词汇表大小
if meta_vocab_size is None:
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")

# 如果dataset没有自带，则定义为50304
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)


# 混合精度，使用梯度缩放加快训练速度，AMP = Automatic Mixed Precision，如果dtype设成float16才会启用。
# 如果前向传播对于一个特定的操作的输入是 float16 数据类型的，那么该操作对应的反向传播也会产生 float16 的梯度。
# 小幅值的梯度值可能在半精度下是不可表示的。这些值可能会刷新为零 (称为underflow)，因此对应参数的更新也会丢失。
# (这里可能是指类似梯度消失的问题，半精度下，如果梯度幅值小，权重在更新时几乎没有改变。)
# 为了避免 underflow，采用梯度缩放(gradient scaling)的方式，将网络的损失乘以一个缩放因子，并对缩放后的损失调用反向传播，
# 然后，通过网络反向流动的梯度将按相同的因子进行缩放。也就是说，缩放后梯度值会有一个较大的幅值，因此它们不会被刷新为零。
# 每个参数的梯度在优化器 (optimizer) 更新参数之前应该是未缩放的，所以缩放因子不会影响学习率。
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
checkpoint = None

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # 新功能，加速训练，需pytorch 2.0+，Cuda

# 记录损失
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Schedule learning rate
def get_lr(it):
    # 当iter数小于warmup步数时，执行线性warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 当执行步数大于学习率衰减步数时，使用最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 当步数在warmup步数和学习率衰减步数之间时，执行余弦退火构建新的学习率  2000 < it < 600000
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1 # 衰减率应为0~1的小数
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# training loop
X, Y = get_batch('train') # 用训练的方式获取数据
t0 = time.time()
local_iter_num = 0 # 初始化iter num
while True:
    # 训练开始前，查看是否有设置学习率衰减，若是执行上面的Schedule learning rate方法，否则使用默认学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 每20步计算一次训练集和验证集损失，并写入checkpoint（没有GPU,非常慢）
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    # 查看是否只是evaluate模式，是则退出训练过程
    if iter_num == 0 and eval_only:
        break

    # 梯度累加
    # 如果显存不足，我们可以通过gradient_accumulation_steps梯度累计来解决。
    # 假设原来的batch size=10,数据总量为1000，那么一共需要100train steps，同时一共进行100次梯度更新。
    # 若是显存不够，我们需要减小batch size，我们设置gradient_accumulation_steps=2，那么我们新的batch size=10/2=5。
    # 我们需要运行两次，才能在内存中放入10条数据，梯度更新的次数不变为100次，那么我们的train steps=200
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # 如果是fp16半精度才会产生梯度缩放的功能
        scaler.scale(loss).backward()

    # 梯度裁剪
    if grad_clip != 0.0: #max norm
        scaler.unscale_(optimizer) # 使用unscale_()取消梯度的缩放，以使得裁剪没有缩放的梯度:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 常规训练步骤
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # 时间更新
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    # 达到最大iter num退出循环
    if iter_num > max_iters:
        break
