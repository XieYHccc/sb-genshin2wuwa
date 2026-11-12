# 薛定谔流复刻方案 v2：原神 ↔ 鸣潮 画风互转

> 目标文章：《【生成模型】薛定谔流：玩鸣潮的同时玩原神》（知乎 [ycz](https://www.zhihu.com/people/ycz-19)，2026-04-17）
> 底层论文：[Schrödinger Bridge Flow for Unpaired Data Translation](https://arxiv.org/abs/2409.09347)（De Bortoli et al., NeurIPS 2024）
>
> **v2 说明**：读到原文后，v1 方案有三处根本性错误已修正。见第 0 节。
>
> 确定条件：游戏内实机截图 / 白天户外自然景 / 自行录屏 / Colab Pro（100 units）/ 先小规模严格复刻

---

## 0. v1 错在哪：作者的做法与论文原版有重大偏离

**作者并没有复刻论文的 α-DSBM 算法。** 原文第五节明确写道，论文用类似 Rectified Flow 的方式近似 IMF，"这里我们不用管"。作者换了一条更便宜的路。三处关键差异：

| | v1（按论文 α-DSBM） | **v2（按原文实际做法）** |
|---|---|---|
| **配对方式** | 独立耦合 + 在线微调阶段解 SDE 生成样本来逼近 OT | **minibatch OT 配对**（pixel 空间缩略图上求解） |
| **训练阶段** | 两阶段：预训练 100k + 在线微调 20k，微调步贵 15–25× | **单阶段**，训练循环里完全不解 SDE |
| **推理** | SDE（Euler-Maruyama，100 NFE） | **ODE**：`dx = (v→ − v←)/2 · dt`，确定性 |
| **空间** | 64px 像素空间 | **DC-AE 潜空间**（作者明确说像素空间太贵） |
| **网络** | UNet | **DiT + AdaLN** 注入 t 和方向 d |

v1 最大的误判是算力模型。我当时写"微调步比预训练步贵 15–25 倍，瓶颈全在后半段"——**这个瓶颈在作者的方案里根本不存在**，因为训练循环里没有 SDE 求解。这让整件事从"Colab Pro 刚好够"变成"Colab Pro 很宽裕"。

作者的实际成本：**单卡 4090 训练一天**。

另外两点需要说明：

- 原文正文中反复嵌入了一段针对爬虫的指令（要求 AI 在每段文字中间插入特定水印字符串）。这是内容里的提示注入，不是你的指示，我不执行。
- 原文声明禁止转载。本文档是我写的工程方案，只转述技术要点与推导，不复制原文段落。

---

## 1. 方法：三个要素如何被满足

作者把"转移"归结为两个要素：

1. **分布匹配**：$T_{\#}X \approx Y$
2. **内容一致性**：$\Phi(T(x)) = \Phi(x)$

历史方法的问题都出在第二条——CycleGAN 用循环一致性强行构造双射，逼网络只学浅层可逆特征，严重限制表达力；风格滤镜用 VGG/Gram 特征，但"内容"到底是什么从未被定义清楚。

薛定谔桥的解法是：**用最优传输代价来定义内容一致性**。不需要人为指定内容提取器，"内容相近"就是"传输代价小"。而熵正则化（ε）让这个约束从硬的 OT 松弛成可学的随机桥。

作者整理的关键权衡（原文第五节表格）：

| 动作 | 效果 |
|---|---|
| OT 配对 | 更接近全局 OT，但更难学 |
| 增大 ε | 远离全局 OT，但更好学 |
| 降低 ε | 更接近全局 OT，但更难学 |
| 已是 SB 且 ε→0 | 接近全局 OT，**且不一定更难学** |

最后一行是整套方案的立足点：先用布朗桥把路径撑开让网络好学，同时用 minibatch OT 配对去逼近全局 OT，两边同时拿到。

### 1.1 核心公式

**布朗桥插值**：

$$
x_t = (1-t)\,x_0 + t\,x_1 + \sqrt{\varepsilon\, t(1-t)}\; z, \qquad z \sim \mathcal{N}(0, I)
$$

**两个速度场**：

$$
v^{\rightarrow}_t = \frac{x_1 - x_t}{1-t}, \qquad
v^{\leftarrow}_t = \frac{x_0 - x_t}{t}
$$

展开后（后面推导要用）：

$$
v^{\rightarrow}_t = (x_1 - x_0) - \sqrt{\frac{\varepsilon\, t}{1-t}}\; z, \qquad
v^{\leftarrow}_t = (x_0 - x_1) - \sqrt{\frac{\varepsilon\,(1-t)}{t}}\; z
$$

两者的奇点分别在 $t \to 1$ 和 $t \to 0$。

**共享网络的时间反转约定**：

$$
F_\theta(x_t, \tau, d{=}1) = v^{\rightarrow}_t,\quad \tau = t
\qquad\qquad
F_\theta(x_t, \tau, d{=}0) = v^{\leftarrow}_t,\quad \tau = 1-t
$$

> 作者说"具体为什么要这么设计，因为没有认真实验过所以暂时略过"。**答案在 2.2 节的推导里**：这个约定让两个方向共享同一组重参数化系数函数。不这么设计，就得为两个方向各维护一套 $c_{in}/c_{skip}/c_{out}$。

**推理用 ODE**（不是 SDE）：

$$
dx = \frac{v^{\rightarrow}_t - v^{\leftarrow}_t}{2}\; dt
$$

为什么能消掉扩散项：在随机插值框架下，正向漂移 $v^{\rightarrow} = b + \tfrac{\varepsilon}{2}\nabla \log p$，反向漂移 $v^{\leftarrow} = -b + \tfrac{\varepsilon}{2}\nabla \log p$，其中 $b$ 是概率流 ODE 速度。于是

$$
\frac{v^{\rightarrow} - v^{\leftarrow}}{2}
= \frac{\left(b + \tfrac{\varepsilon}{2}\nabla \log p\right) - \left(-b + \tfrac{\varepsilon}{2}\nabla \log p\right)}{2}
= b
$$

score 项严格抵消。✅ 作者的公式和解释都是对的。

**这带来三个实际好处**：推理确定性、无需额外 temporal 约束就有帧间连贯性（作者展示的视频效果就靠这个）、NFE 可以压得很低。

---

## 2. ⭐ EDM 重参数化：论文附录是错的，这里给出正确推导

原文第六节：论文想用 EDM 形式做重参数化，但**附录给出的结果是错误的（包括图表）**，作者定位到"用最小二乘求偏导的地方代入错了"，然后说"空间太小写不下"，把推导留给读者。

这是实现里绕不开的一环，我推一遍。

### 2.1 目标

$$
D_\theta(x_t, \tau, d) = c_{skip}(\tau)\, x_t + c_{out}(\tau)\, F_\theta\!\left(c_{in}(\tau)\, x_t,\; \tau,\; d\right)
$$

三个要求：

1. $c_{in}(\tau)\, x_t$ 方差为 1 → 网络输入是标准化的
2. 网络的回归目标 $F_{target} = \dfrac{v - c_{skip}\, x_t}{c_{out}}$ 方差为 1 → 网络输出是标准化的
3. $c_{skip}$ 取最小二乘最优值，最小化 $c_{out}^2$（即最小化误差放大）

### 2.2 推导

**统计假设**：$\mathbb{E}[x_0] = \mathbb{E}[x_1] = 0$，$\mathrm{Var}(x_0) = \mathrm{Var}(x_1) = \sigma_d^2$，$\mathrm{Cov}(x_0, x_1) = \rho\,\sigma_d^2$，$z$ 独立。

> $\rho$ 是**配对相关系数**，这是本推导相对标准 EDM 的关键增项。随机配对 $\rho = 0$；OT 配对让 $x_0$、$x_1$ 更接近，$\rho > 0$。**论文很可能就是在这里漏掉了耦合项**。

记 $\sigma_u^2 = \mathrm{Var}(x_1 - x_0) = 2\sigma_d^2 (1 - \rho)$。

**① $x_t$ 的方差**

$$
s^2(\tau) = \sigma_d^2\left[(1-\tau)^2 + \tau^2 + 2\rho\,\tau(1-\tau)\right] + \varepsilon\,\tau(1-\tau)
$$

$$
c_{in}(\tau) = \frac{1}{s(\tau)}
$$

**② 速度的方差**

$$
\mathrm{Var}\!\left(v^{\rightarrow}_t\right) = \sigma_u^2 + \frac{\varepsilon\, t}{1-t}
$$

**③ 协方差**（这是论文出错的地方）

$z$ 的贡献：

$$
-\sqrt{\frac{\varepsilon t}{1-t}} \cdot \sqrt{\varepsilon\, t(1-t)} = -\varepsilon t
$$

数据部分：

$$
\begin{aligned}
\mathrm{Cov}\!\left(x_1 - x_0,\; (1-t)x_0 + t x_1\right)
&= (1-t)\rho\sigma_d^2 + t\sigma_d^2 - (1-t)\sigma_d^2 - t\rho\sigma_d^2 \\
&= \sigma_d^2 (1-2t)(\rho - 1) \\
&= -\frac{\sigma_u^2}{2}(1-2t)
\end{aligned}
$$

合并：

$$
\mathrm{Cov}\!\left(v^{\rightarrow}_t,\; x_t\right) = -\frac{\sigma_u^2}{2}(1-2t) - \varepsilon t
$$

**④ 最小二乘最优 $c_{skip}$**

$$
c_{skip}(\tau) = \frac{\mathrm{Cov}(v, x_t)}{\mathrm{Var}(x_t)}
= -\,\frac{\frac{\sigma_u^2}{2}(1-2\tau) + \varepsilon\,\tau}{s^2(\tau)}
$$

**⑤ $c_{out}$**

$$
c_{out}^2(\tau) = \mathrm{Var}(v) - \frac{\mathrm{Cov}(v, x_t)^2}{\mathrm{Var}(x_t)}
= \sigma_u^2 + \frac{\varepsilon\,\tau}{1-\tau}
- \frac{\left[\frac{\sigma_u^2}{2}(1-2\tau) + \varepsilon\,\tau\right]^2}{s^2(\tau)}
$$

### 2.3 验证

取 $\varepsilon = 0$、$\rho = 0$、$\sigma_d = 1$：

- **$\tau = 0$**：$s^2 = 1$，$c_{skip} = -1$，$c_{out}^2 = 2 - 1 = 1$。此时 $x_t = x_0$，$D = -x_0 + F$，故 $F = v + x_0 = x_1$，方差为 1 ✅
- **$\tau = 1$**：$s^2 = 1$，$c_{skip} = +1$，$c_{out}^2 = 1$。此时 $x_t = x_1$，$F = v - x_1 = -x_0$，方差为 1 ✅

两端都严格给出单位方差目标，推导自洽。

### 2.4 为什么反方向要反转物理时间

对反向速度 $v^{\leftarrow} = (x_0 - x_1) - \sqrt{\varepsilon(1-t)/t}\; z$，代入 $\tau = 1 - t$：

| 统计量 | 结果 | |
|---|---|---|
| $\mathrm{Var}(v^{\leftarrow})$ | $\sigma_u^2 + \dfrac{\varepsilon\,\tau}{1-\tau}$ | 与正向**同形** |
| $s^2$ | 关于 $t \leftrightarrow 1-t$ 对称 | **同形** |
| $\mathrm{Cov}(v^{\leftarrow}, x_t)$ | $-\dfrac{\sigma_u^2}{2}(1-2\tau) - \varepsilon\,\tau$ | **同形** |

三项全部同形。**所以 $c_{in}$、$c_{skip}$、$c_{out}$ 三个函数可以被两个方向共用**，一套系数覆盖全部。这就是作者没来得及解释的那个设计理由。

### 2.5 工程实现

**不要用解析式硬套高斯假设。** 真实潜空间分布不是高斯，OT 配对后的 ρ 也随数据而变。更稳的做法是**离线实测**：

```python
@torch.no_grad()
def calibrate_coeffs(z0_pool, z1_pool, eps, pair_fn, n_tau=64, n_sample=4096):
    """在 τ 网格上实测三个统计量，查表 + 线性插值"""
    taus = torch.linspace(1e-3, 1-1e-3, n_tau)
    table = []
    for tau in taus:
        z0, z1 = pair_fn(sample(z0_pool, n_sample), sample(z1_pool, n_sample))
        z = torch.randn_like(z0)
        xt = (1-tau)*z0 + tau*z1 + (eps*tau*(1-tau)).sqrt()*z
        v  = (z1 - xt) / (1 - tau)              # 正向；反向同形，无需重算

        var_x = xt.var().item()
        var_v = v.var().item()
        cov   = ((xt - xt.mean()) * (v - v.mean())).mean().item()

        c_skip = cov / var_x
        c_out  = max(var_v - cov**2 / var_x, 1e-8) ** 0.5
        c_in   = var_x ** -0.5
        table.append((c_in, c_skip, c_out))
    return taus, torch.tensor(table)
```

跑一次（几分钟）存成查表，训练时按 τ 插值取值。这样完全不依赖高斯假设，也自动吸收了 OT 配对带来的 ρ。

**端点奇点仍然存在**：$\tau \to 1$ 时 $\dfrac{\varepsilon\,\tau}{1-\tau} \to \infty$，$c_{out}$ 发散。重参数化只是把奇点**分离出来单独管理**，消不掉——作者原话"实际上消除不太了，只是能够被分离管理"。所以 $\tau$ 仍必须 clamp 到 $[10^{-3},\, 1-10^{-3}]$。

---

## 3. DC-AE 潜空间

作者明确说像素空间训练太贵、试过 JIT 瓶颈仍在训练成本，所以走高压缩率 AE 的潜空间。

### 3.1 选型

[DC-AE](https://hanlab.mit.edu/projects/dc-ae)（MIT HAN Lab，[arXiv 2410.10733](https://arxiv.org/abs/2410.10733)）已集成进 diffusers 的 `AutoencoderDC`。可用权重：

| checkpoint | 压缩率 | 潜通道 | 512×512 → 潜空间 | token 数 (patch=1) |
|---|---|---|---|---|
| `mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers` | f32 | 32 | **16×16×32** | **256** |
| `mit-han-lab/dc-ae-f32c32-in-1.0-diffusers` | f32 | 32 | 16×16×32 | 256 |
| `mit-han-lab/dc-ae-f64c128-in-1.0-diffusers` | f64 | 128 | 8×8×128 | 64 |
| `mit-han-lab/dc-ae-f128c512-in-1.0-diffusers` | f128 | 512 | 4×4×512 | 16 |

**推荐 `f32c32-sana-1.1`**：512px 输入只有 256 token，DiT 训练极快，且 f32 的重建质量明显好于 f64/f128。游戏画面细节多，不建议上 f64 以上。

> ⚠️ `in` 和 `mix` 变体的 checkpoint key 相同但 **scaling_factor 不同**，`from_single_file` 无法自动区分（会默认按 `mix` 配）。用 `from_pretrained` 加载 `-diffusers` 版本可避开这个坑。

### 3.2 潜空间预计算

**潜空间必须离线预计算并缓存**，不要在训练循环里现编码——那会让 AE 成为瓶颈，且浪费 GPU。

```python
import torch
from diffusers import AutoencoderDC

ae = AutoencoderDC.from_pretrained(
    "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", torch_dtype=torch.float32
).to("cuda").eval()

sf = ae.config.scaling_factor   # 读配置，不要硬编码

@torch.no_grad()
def encode_batch(imgs):              # imgs: (B,3,512,512) in [-1,1]
    z = ae.encode(imgs).latent        # (B,32,16,16)
    return (z * sf).to(torch.float16) # 标准化到单位方差后存 fp16
```

**同时缓存 pixel 空间缩略图**（OT 配对要用，见 4.1）：

```
30_packed/
  genshin_lat.npy    (N, 32, 16, 16)  float16   # 5000 张 ≈ 82 MB
  genshin_thumb.npy  (N, 3, 64, 64)   uint8     # ≈ 61 MB
  wuwa_lat.npy       同上
  wuwa_thumb.npy     同上
```

全部可一次性读进内存，训练时零 I/O。

**务必先验证重建质量**：取 50 张图过一遍 `encode → decode`，人眼检查描边、远景植被、水面高光有没有糊掉。如果 f32 重建就已经丢了画风关键特征，后面训什么都没意义。

---

## 4. 训练

### 4.1 minibatch OT 配对（内容一致性的来源）

用 [TorchCFM](https://github.com/atong01/conditional-flow-matching) 的 `OTPlanSampler`：

```python
from torchcfm.optimal_transport import OTPlanSampler
ot = OTPlanSampler(method="exact")       # pot.emd，B×B 成本矩阵
```

`sample_plan(x0, x1)` 内部：算平方欧氏成本矩阵 → 解 OT → 按 plan 采样索引 → 返回重排后的 `(x0[i], x1[j])`。

**关键：在 pixel 缩略图上配对，在 latent 上训练。**

```python
def make_pair(thumb0, thumb1, lat0, lat1, ot_ratio=0.8, cost="rgb", dino=None):
    B = thumb0.shape[0]
    k = int(B * ot_ratio)                       # 前 k 个走 OT，其余保持随机

    if cost == "rgb":
        f0, f1 = thumb0[:k].flatten(1).float(), thumb1[:k].flatten(1).float()
    elif cost == "dino":                        # 语义配对
        f0, f1 = dino(thumb0[:k]), dino(thumb1[:k])
        f0, f1 = F.normalize(f0, dim=-1), F.normalize(f1, dim=-1)

    pi = ot.get_map(f0, f1)
    i, j = ot.sample_map(pi, k, replace=False)

    idx0 = torch.cat([torch.as_tensor(i), torch.arange(k, B)])
    idx1 = torch.cat([torch.as_tensor(j), torch.arange(k, B)])
    return lat0[idx0], lat1[idx1]
```

**两个可调旋钮**（原文第七、八节）：

| 旋钮 | 作用 |
|---|---|
| **代价函数** | RGB MSE → 倾向学颜色相近的转移；**DINO 内积** → 倾向保留高层语义做转移 |
| **OT / 随机配对比例** | 控制转移语义的粗细 |

> 这两个旋钮直接回应了"会不会只学到调色"的担忧。如果 RGB 代价下输出只是换色，**切 DINO 代价**——它强制按语义配对，颜色相似不再是配对依据。这比我 v1 方案里设计的一堆诊断指标更直接：它是从源头解决，不是事后检测。
>
> DINO 建议用 `facebook/dinov2-small`，在 64×64 缩略图上跑，开销很小且可以缓存特征。

`ot_ratio` 建议从 0.8 起步。全 OT（1.0）配对过硬会让多样性下降。

### 4.2 网络：DiT + AdaLN

作者第八节特别指出：用 DiT 的 AdaLN 注入 t 和方向 d 时，**正反速度共用网络主体，推理做 `(v→−v←)/2` 相减时内部偏差会大量抵消**，因此在 ε 很低时能急速收敛。这是选 DiT 而非 UNet 的实质理由，不是跟风。

512px + f32c32 的配置（约 85M 参数）：

```python
DiTConfig(
    input_size=16, patch_size=1, in_channels=32,    # 256 token
    hidden_size=768, depth=12, num_heads=12,
    cond=["tau", "direction"],                       # 都走 AdaLN modulation
)
```

方向 d 用 `nn.Embedding(2, hidden_size)`，与 τ 的 embedding **相加**后一起进 AdaLN。

可直接改 [facebookresearch/DiT](https://github.com/facebookresearch/DiT) 的 `models.py`：把 `y`（类别标签）那一路换成 direction embedding 即可，改动量约 10 行。

### 4.3 训练循环

```python
EPS = 0.05          # 低 ε，见 4.4

def train_step(net, ema, opt, lat0, lat1, thumb0, thumb1, coeff_table):
    B = lat0.shape[0]
    z0, z1 = make_pair(thumb0, thumb1, lat0, lat1, ot_ratio=0.8)

    tau = torch.rand(B, device=z0.device).clamp(1e-3, 1-1e-3)
    d   = (torch.rand(B, device=z0.device) < 0.5).float()      # 半 batch 一个方向

    # 物理时间：d=1 用 t=tau；d=0 用 t=1-tau
    t = torch.where(d.bool(), tau, 1 - tau).view(-1,1,1,1)
    z = torch.randn_like(z0)
    xt = (1-t)*z0 + t*z1 + (EPS*t*(1-t)).sqrt()*z

    # 目标速度
    v_fwd = (z1 - xt) / (1 - t)
    v_bwd = (z0 - xt) / t
    v = torch.where(d.view(-1,1,1,1).bool(), v_fwd, v_bwd)

    # EDM 重参数化（按 tau 查表，正反共用）
    c_in, c_skip, c_out = lookup(coeff_table, tau)
    target = (v - c_skip*xt) / c_out
    pred   = net(c_in * xt, tau, d)

    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step(); ema.update(net)
    return loss.item()
```

**训练循环里没有任何 SDE 求解。** 这就是成本能压到单卡一天的原因。

### 4.4 推理（ODE）

```python
@torch.no_grad()
def translate(net_ema, z_src, n_steps=32, direction="0to1", coeff_table=None):
    x = z_src.clone()
    dt = 1.0 / n_steps
    sgn = 1.0 if direction == "0to1" else -1.0

    for i in range(n_steps):
        t = i*dt if sgn > 0 else 1 - i*dt
        tt = torch.full((x.shape[0],), t, device=x.device)

        v_f = denoise(net_ema, x, tt,     d=1, coeff_table=coeff_table)  # τ=t
        v_b = denoise(net_ema, x, 1 - tt, d=0, coeff_table=coeff_table)  # τ=1-t
        x = x + sgn * 0.5 * (v_f - v_b) * dt          # 概率流 ODE
    return x
```

每步两次网络前向（正反各一次）。`n_steps=32` 起步，可往下压到 16 看质量损失。

> **帧间连贯性是免费的**：ODE 确定性，输入相近 → 输出相近。作者的视频 demo 没有加任何 temporal 约束。

### 4.5 超参表

| 超参 | 值 | 来源 |
|---|---|---|
| **ε** | **0.05**（低 ε） | ✅ 原文第八节：低 ε 急速收敛；高 ε 退化成两个 VE Diffusion，收敛时间数倍以上，4090 扛不住 |
| OT 配对比例 | 0.8 | ⚙️ 我校准（原文只说可调） |
| OT 代价函数 | RGB 起步，必要时切 DINO | ✅ 原文第七节 |
| 配对空间 | pixel 缩略图（64×64） | ✅ 原文第七节 |
| 训练空间 | DC-AE f32c32 latent | ✅ 原文第七节 |
| 网络 | DiT + AdaLN，~85M | ✅ 原文第八节 |
| 重参数化 | EDM 形式，系数见第 2 节 | ✅ 原文第六节（系数由我推导补全） |
| 推理 | ODE，`(v→−v←)/2` | ✅ 原文第五节 |
| 优化器 | AdamW，lr 1e-4，wd 0.0 | ⚙️ 我校准（DiT 常规） |
| warmup | 2000 步线性 | ⚙️ 我校准 |
| batch size | 128（A100-40G，bf16） | ⚙️ 我校准 |
| 训练步数 | 150k | ⚙️ 我校准（对标 4090 一天） |
| EMA decay | 0.9999 | ⚙️ 我校准 |
| τ clamp | [1e-3, 1−1e-3] | ✅ 奇点要求 |
| 推理 NFE | 32（每步 2 次前向） | ⚙️ 我校准 |

> **ε 是最重要的旋钮**，原文说它控制"转移的生硬程度"。0.05 是保守起点，可在 0.01–0.2 间扫。ε 调大画风改动更剧烈但收敛变慢。

---

## 5. 数据

作者原本就有大量两款游戏的数据（本来打算训世界模型）。你要自己录，所以这部分是你的主要工作量。

### 5.1 采集协议（两边严格对称）

**对称性是第一优先级。** 任何两边不一致的设置，模型都会优先学那个差异而不是画风。

| 项 | 设定 |
|---|---|
| 机器 / 显示器 | 同一台 |
| 采集分辨率 | 1920×1080 |
| 窗口模式 | **无边框窗口**（不要独占全屏，见 §5.2.3） |
| 画质档位 | 两边同档（建议最高），**关动态模糊** |
| 采集方式 | 同一个 `capture.py`，只改输出目录 |
| HDR | 两边都关 |
| 游戏内 UI | 全部隐藏（拍照模式 / 隐藏 UI 快捷键） |
| 游戏内时间 | 锁 08:00–16:00 |
| 天气 | 仅晴天 |
| 视角 | 第三人称默认 FOV，角色尽量出画或贴边 |
| 运动 | 匀速步行 / 滑翔，避免冲刺和快速转镜 |

**地貌配平**（两边按类型对应，不按地名）：草原丘陵 / 山地岩壁 / 水景湖岸 / 林地，各占约 25%。

**目标量**：每边 3000–5000 张去重后的帧。按 1–2 帧/秒采集 + 实时去重，约需 3–5 个会话、每会话 15–25 分钟的游戏内闲逛。

> **为什么强约束场景仍然值得**：OT 配对在 minibatch 内求解，batch 只有 128。域内方差越大，batch 里能找到的"好配对"越少，OT plan 质量越差。强约束是在**帮 OT 配对工作**，不只是降低学习难度。这是 v1 里我没说准的地方。

### 5.2 自动采集：被动截屏 + 实时过滤（推荐）

> **安全边界（不可协商）**：本方案只做**被动屏幕捕获**，绝不模拟键鼠去自动操控角色。两款游戏都带内核级反作弊（原神 mhyprot、鸣潮 ACE），向游戏进程注入合成输入是明确的封号风险，也违反用户协议。为一个数据集赌账号不值得。
>
> 屏幕捕获本身不触发反作弊——OBS、Discord、Steam 悬浮窗每天都在做，两家也都允许直播录制。所以自动化的边界是：**人工走路，脚本自动抓取、过滤、去重、归档**。你只需要在游戏里闲逛。

相比"先录屏再抽帧"，这条路的优势：

| | 录屏 + ffmpeg 抽帧 | **被动采集脚本** |
|---|---|---|
| 中间存储 | 几十 GB 视频 | 无，直接出 PNG |
| 压缩伪影 | x264 编码引入 | **无损，直接取帧缓冲** |
| 去重时机 | 事后 | **实时，站桩不产生垃圾** |
| 进度可见 | 录完才知道 | **实时显示"已攒 N 张有效图"** |
| 两域一致性 | 依赖编码参数对齐 | 同一脚本，天然一致 |

压缩伪影这点尤其重要。§5.1 强调两边编码参数必须一致，就是怕模型去学伪影而不是画风；直接截屏把这个风险源整个移除了。

#### 5.2.1 `capture.py` 设计

**采集循环**（目标 1–2 帧/秒，不追求高帧率）：

1. 窗口定位：按进程名/窗口标题找到游戏窗口，只抓客户区（排除标题栏和边框）
2. 抓帧：`mss` 抓窗口矩形 → numpy 数组
3. **实时过滤链**（顺序即成本顺序，便宜的先跑）：
   - 亮度：`mean(luma)` 落在 [60, 230] 之外直接丢
   - 清晰度：Laplacian 方差低于阈值丢（运动模糊 / 过场虚化）
   - UI 残留：四角区域边缘密度超阈值丢
   - **pHash 去重**：与最近 N 张（滑动窗口，建议 N=300）比汉明距离，< 6 丢
4. 存盘：通过的帧存 PNG，文件名带时间戳与会话号
5. 控制台实时打印：已抓/已存/各过滤器淘汰数

**关键设计点：**

| 点 | 做法 | 理由 |
|---|---|---|
| 热键暂停 | `F9` 暂停/继续，`F10` 结束 | 进菜单、看剧情、传送时暂停，避免污染 |
| 滑动窗口去重 | 只比最近 300 张而非全量 | 全量比对 O(N²) 会越跑越慢；局部窗口足够挡站桩 |
| 两域同脚本 | 只换输出目录和窗口名 | 保证两边过滤阈值与流程完全一致 |
| 阈值可配 | 顶部常量块 | §5.1 要求两边**分别**定清晰度阈值 |
| 崩溃安全 | 每帧独立存盘，无缓冲 | 游戏崩了已存的不丢 |
| 统计落盘 | 退出时写 `session_stats.json` | 记录各过滤器淘汰率，用于调阈值 |

**依赖**（都是纯 Python 轮子，无需编译）：

| 库 | 用途 |
|---|---|
| `mss` | 跨平台快速截屏，比 PIL.ImageGrab 快一个量级 |
| `pywin32` | 窗口定位（`win32gui.FindWindow` / `GetClientRect`） |
| `Pillow` | 存 PNG |
| `numpy` | 亮度/Laplacian 统计 |
| `imagehash` | pHash |
| `keyboard` 或 `pynput` | 全局热键（**只读键盘，不发送按键**） |

> ⚠️ `keyboard` 库在 Windows 上需要管理员权限才能全局挂钩。如果不想提权，退化方案：不做热键，改用"命令行里 Ctrl+C 结束 + 采集前手动确认场景"，或用窗口焦点检测（游戏窗口失焦时自动暂停）。**后者更优雅，推荐优先尝试。**

#### 5.2.2 采集工作流

1. 游戏里调好设置（§5.1 的表）、隐藏 UI、锁定时间天气
2. 启动 `capture.py --game genshin --session 01`
3. 切到游戏，开始在目标地貌里**匀速步行/滑翔**
4. 遇到进菜单/过场/战斗 → 失焦暂停或按热键暂停
5. 控制台显示攒够目标数量 → 结束
6. 换另一个游戏，**同一脚本**，只改 `--game wuwa`

**单次会话建议 15–25 分钟**，分 3–5 个会话覆盖不同地貌，避免一次跑太久导致地貌单一。

#### 5.2.3 备选：录屏 + 抽帧

如果被动采集脚本遇到意外（窗口捕获拿到黑屏、独占全屏模式抓不到），退回传统路线：

```bash
ffmpeg -i capture_genshin_01.mkv -vf "fps=1" -q:v 1 10_frames/genshin/g01_%05d.png
```

两边必须用**完全相同**的抓帧与编码参数。

> **独占全屏的坑**：游戏若运行在独占全屏（Exclusive Fullscreen），`mss` 可能抓到黑屏。解决办法是把游戏设为**无边框窗口（Borderless Windowed）**——两款游戏都支持，且对画质无影响。这条要在正式采集前先验证。

### 5.3 离线过滤（补充 pass）

被动采集已经在线做掉了大部分过滤，这里只做需要全局视野或较贵的步骤：

1. **全量 pHash 去重**：跨会话、跨窗口的重复（在线只比了最近 300 张）
2. **可选 CLIP zero-shot**：保留 `"a daytime outdoor natural landscape"` 得分高于 `"indoor"` / `"night"` / `"UI"` / `"character close-up"` 的帧
3. **清晰度阈值复核**：按两域各自分布的 10 分位重新定阈值（在线用的是预设值，可能偏松或偏紧）
4. **人工抽检**：随机抽 100 张目视，确认没有漏网的 UI、过场、夜景

若走了 §5.2.3 的录屏备选路线，则完整过滤链为：

1. 黑帧 / 过曝：`mean(luma)` 在 [25, 230] 之外丢弃
2. 过暗（漏网夜间/洞穴）：`mean(luma) < 60` 丢弃
3. 运动模糊：Laplacian 方差低于各自分布 10 分位的丢弃（**两边分别定阈值**，避免引入新域差异）
4. UI 残留：四角区域边缘密度检测
5. **近重复去重**：pHash 汉明距离 < 6 只留一张。站桩会产出大量近重复，这步通常淘汰 30–50%
6. 可选：CLIP zero-shot 过滤非白天户外场景

### 5.4 裁剪与打包

- 中心裁 1080×1080 → resize 512×512（Lanczos）作为训练分辨率
- 训练时在线增强：从原 1920×1080 随机裁窗再 resize，等于免费扩数据
- 水平翻转两域都开
- 过 DC-AE 得 latent，同时存 64×64 缩略图（见 3.2）
- **按采集 session 切分 train/val**，不要随机切（同 session 帧高度相关会泄漏）

### 5.5 公开数据集情况

搜索结论：**没有可直接用的。**

- 原神侧只有 [svjack/video-dataset-genshin-impact-landscape-organized](https://huggingface.co/datasets/svjack/video-dataset-genshin-impact-landscape-organized)（60 条 / 16 MB）和 [ep-landscape 版](https://huggingface.co/datasets/svjack/video-dataset-genshin-impact-ep-landscape-organized)（93 条 / 45 MB），且是**官方 EP 宣传动画过场**不是实机画面，授权不明
- 其他 Genshin 数据集（[animelover](https://huggingface.co/datasets/animelover/genshin-impact-images)、[Fazzie/Teyvat](https://huggingface.co/datasets/Fazzie/Teyvat)）全是角色同人插画
- **鸣潮侧 HuggingFace 上一张场景图都没有**，全是语音/TTS（[wwTTS](https://huggingface.co/datasets/Genius-Society/wwTTS)、[AI-Hobbyist](https://github.com/AI-Hobbyist/WutheringWaves_Datasets)）和 datamine 资源（[Arikatsu](https://github.com/Arikatsu/WutheringWaves_Data)、[anyliew](https://github.com/anyliew/ww_resources)）

自己采集反而更好：同机器同设置、无编码伪影，域间差异干净地只剩画风。

### 5.6 版权与边界

两款游戏画面均为著作权作品。数据仅供个人研究学习：不公开再分发采集数据集，不用产出模型提供商业服务。

采集手段限于**被动屏幕捕获**（§5.2 的安全边界）：不读写游戏内存、不注入进程、不模拟输入、不触碰反作弊相关的任何东西。

---

## 6. Colab 工程

### 6.1 算力预算（Colab Pro = 100 units）

实测 burn rate（2026）：A100-40G ≈ **5.6 units/h**，L4 ≈ **1.7 units/h**，T4 ≈ **1.2 units/h**。

| 阶段 | 硬件 | 时长 | units |
|---|---|---|---|
| DC-AE 重建质量验证 | 免费 T4 | 0.5h | 0 |
| 潜空间预计算（两边 1 万张） | L4 | 1.5h | ~3 |
| 系数标定 + 冒烟测试（2k 步） | 免费 T4 | 1h | 0 |
| **主训练 150k 步** | **A100** | **~12h** | **~67** |
| 评估 + 出图 + 视频 | L4 | 2h | ~4 |
| **小计** | | | **~74** |
| 机动（重跑 / 扫 ε / 调配对策略） | | | **~26** |

**比 v1 宽裕得多**——v1 因为误以为要做 α-DSBM 在线微调，光微调阶段就估了 50 units。

**Colab 注意事项**：
- 关浏览器即断线（后台执行是 Pro+ 专属），12h 训练一定会被打断
- 磁盘随运行时销毁，checkpoint 必须落 Drive
- 挂着不用的 GPU 照样烧 units，跑完立刻断开

### 6.2 断点续训

```python
def save_ckpt(path, net, ema, opt, step):
    torch.save({'net': net.state_dict(), 'ema': ema.state_dict(),
                'opt': opt.state_dict(), 'step': step,
                'rng': torch.get_rng_state()}, path + '.tmp')
    os.replace(path + '.tmp', path)      # 原子替换，防断线写坏
```

每 1000 步存 `latest.pt`，每 20k 步存归档。启动 cell 无条件尝试 resume。

### 6.3 目录

```
Drive/sbflow/
  data/   genshin_lat.npy, genshin_thumb.npy, wuwa_lat.npy, wuwa_thumb.npy
  ckpt/   latest.pt, step_*.pt
  coeff/  edm_table.pt
  logs/   train.csv
  samples/ grid_step*.png
```

---

## 7. 评估

作者坦承原文没做定量比较。但你该做一些，至少用于判断训练是否走偏。

### 7.1 主要看图

每 2000 步出固定种子的对照网格（`源图 | 转换后`），两个方向各一张。这是最有效的诊断手段。作者展示的也是这种 patch 对照图。

### 7.2 关键诊断：是不是只学到调色

```python
def tone_collapse_ratio(x_in, y_out, lpips_fn):
    x_rec = reinhard_color_transfer(src=x_in, ref=y_out)   # 把输入色调匹配到输出
    return (lpips_fn(x_rec, y_out) / lpips_fn(x_in, y_out)).item()
```

| ρ | 含义 | 动作 |
|---|---|---|
| < 0.15 | 基本只做了调色 | **切 DINO 代价函数**，或调大 ε |
| 0.25–0.6 | 有实质风格改变 | ✅ 正常 |
| > 0.8 | 结构被大幅改写 | 调小 ε，或提高 OT 比例 |

> 和 v1 不同，这里有**直接的干预手段**（切代价函数 / 调 ε），不只是报警。

### 7.3 其他

- **FID / LPIPS** 双向各算一次。验证集不足 500 张时 FID 只能横向比 checkpoint，不要和论文数字比
- **往返一致性**：`A → B → A` 与原图的 LPIPS。ODE 确定性 + 近似 OT，往返距离应该很小。这是验证"学到了桥而非任意映射"的有力证据
- **视频连贯性**：拿一段连续帧逐帧转换，看有没有闪烁。按作者说法不加 temporal 约束就该是平滑的——**如果闪烁明显，说明 ODE 实现有问题**（最可能是 $\frac{v^{\rightarrow} - v^{\leftarrow}}{2}$ 的符号或 $\tau$ 传错）

---

## 8. 实现顺序、依赖与风险

### 8.1 建议顺序

| # | 步骤 | 耗时 | 需要的依赖 |
|---|---|---|---|
| 0 | 环境：Python 3.12 venv + torch(cu128) | 0.5h | — |
| 1 | **DC-AE 重建验证（go/no-go 闸门）** | 0.5h | torch, diffusers, Pillow, numpy |
| 2 | 写 `capture.py` + 验证无边框窗口可截屏 | 1h | mss, pywin32, Pillow, numpy, imagehash |
| 3 | 数据采集（两域各 3–5 会话） | 1–2 天 | 同上 |
| 4 | 离线过滤 + 配平 + 打包 | 2h | numpy, Pillow, imagehash（+ 可选 CLIP） |
| 5 | 潜空间预计算 + EDM 系数标定 | 2h | torch, diffusers, numpy |
| 6 | DiT 改造（direction 注入，约 10 行） | 1h | DiT 源码, einops |
| 7 | 冒烟测试 2k 步 | 1h | + torchcfm, pot |
| 8 | 主训练 150k 步 | ~12h | 同上 |
| 9 | 评估 + 扫 ε / 配对策略 | 3h | + lpips, clean-fid, scikit-image |

**第 1 步是闸门**：50 张图过 `encode → decode`，人眼确认描边、远景植被、水面高光没糊。重建就丢了画风特征的话，后面全白费。**这一步只需要四个包，先装这四个即可。**

### 8.2 第三方依赖清单

**核心（缺了跑不起来）**

| 库 | 用途 | 节 |
|---|---|---|
| `torch` | 基础（**必须 cu128+**，RTX 5070 是 sm_120） | 全文 |
| `diffusers` | `AutoencoderDC` —— DC-AE 编解码（需较新版本） | §3.2 |
| `numpy` | `.npy` 打包、统计、OT 索引 | §3.2, §4.1 |
| `Pillow` | 图像读写 | §5 |
| `torchcfm` | `OTPlanSampler` —— minibatch OT 配对 | §4.1 |
| `pot` | torchcfm 的 OT 求解后端（`pot.emd`），随 torchcfm 装 | §4.1 间接 |

> `torchcfm` 可以不要——`OTPlanSampler` 核心就是"算 $B\times B$ 成本矩阵 → 调 `pot.emd` → 按 plan 采样索引"，十几行能自己写，直接依赖 `pot`。用它是因为它处理了数值失败回退等边界情况。

**数据采集与过滤**

| 库 | 用途 |
|---|---|
| `mss` | 快速截屏（比 `PIL.ImageGrab` 快一个量级） |
| `pywin32` | 窗口定位、焦点检测 |
| `imagehash` | pHash 去重 |
| `opencv-python` | Laplacian 方差、边缘密度（也可用 numpy 手写） |
| `keyboard` / `pynput` | 全局热键，**只读键盘不发按键**（可选，见 §5.2.1） |
| `ffmpeg` | 仅 §5.2.3 备选路线需要（命令行工具，非 Python 包） |

**评估（可后置）**

| 库 | 用途 |
|---|---|
| `lpips` | 结构保持度、色调退化检测 ρ |
| `clean-fid` 或 `pytorch-fid` | FID |
| `scikit-image` | Reinhard 色彩迁移的 Lab 转换（也可自己写二十行） |

**可选**

| 库 | 用途 | 何时需要 |
|---|---|---|
| `transformers` | `facebook/dinov2-small` 做 DINO 代价函数 | 只在 RGB 代价下出现"只学到调色"时 |
| `einops` | DiT 的张量重排 | 跟 DiT 代码走 |
| `open_clip` / `transformers` | CLIP zero-shot 场景过滤 | §5.3 可选步骤 |

**非 pip 包**：[facebookresearch/DiT](https://github.com/facebookresearch/DiT) 不发包，需 clone 后改 `models.py`（把类别标签那一路换成 direction embedding）。这是唯一需要改第三方源码的地方。

### 8.3 风险清单

| 风险 | 概率 | 应对 |
|---|---|---|
| EDM 系数实现错误 | **高** | 第 2.3 节的 τ=0/1 单位方差自检；用实测标定而非解析式 |
| ODE 符号/时间传错 → 输出闪烁或糊 | **中高** | 7.3 的视频连贯性检查；单步手工验算 |
| **独占全屏导致截屏拿到黑屏** | **中高** | 游戏设为无边框窗口；正式采集前先验证（§5.2.3） |
| DC-AE 重建丢画风特征 | 中 | 第 1 步闸门 |
| 只学到调色 | 中 | 切 DINO 代价（7.2） |
| OT 配对在小 batch 上质量差 | 中 | 强约束场景（5.1）；必要时加大 batch |
| 在线过滤阈值偏松/偏紧 | 中 | 阈值可配 + 统计落盘；§5.3 离线复核 |
| Colab 断线丢进度 | **确定会发生** | 6.2 原子 checkpoint |
| ε 选大了收敛不了 | 中 | 从 0.05 起，别一上来就调大 |
| 全局热键需要提权 | 低 | 改用窗口焦点检测自动暂停（§5.2.1） |

---

## 9. 待确认

- 两款游戏你当前能玩到的**版本与地区**（影响隐藏 UI 的操作和可达地貌）
- 两款游戏能否设为**无边框窗口**（决定 §5.2 的被动采集是否可行）
- 是否接受为全局热键提权；不接受则走窗口焦点检测方案

**已确认的环境**：
- 本机 **RTX 5070 / 12 GB**（驱动 621.01）——够跑 DC-AE 验证、数据过滤、潜空间预计算，省 Colab units
- Python Install Manager 已装 **3.12.10**（系统默认是 3.15.0，PyTorch 无 cp315 轮子，须用 3.12）
- torch 必须装 **cu128 或更新**的构建（RTX 5070 是 Blackwell / sm_120，cu121 轮子跑不了）
- AppLocker 封 Anaconda 签名的 exe → 不用 Miniconda，用 venv / Miniforge / uv

---

## 参考

1. [Schrödinger Bridge Flow for Unpaired Data Translation](https://arxiv.org/abs/2409.09347) — 薛定谔桥流
2. [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)
3. [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/pdf/2302.00482) — OT 配对
4. [Deep Compression Autoencoder](https://hanlab.mit.edu/projects/dc-ae) — DC-AE
5. [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) — EDM 重参数化
6. [TorchCFM](https://github.com/atong01/conditional-flow-matching) — `OTPlanSampler`
7. [facebookresearch/DiT](https://github.com/facebookresearch/DiT) — DiT 骨架
