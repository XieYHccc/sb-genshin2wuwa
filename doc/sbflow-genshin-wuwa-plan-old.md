# 薛定谔流复刻方案 v3：原神 ↔ 鸣潮 画风互转

> **存档**：这是 2026-11-14 重写之前的旧版方案（v3），已被同目录的 `sbflow-genshin-wuwa-plan.md` 取代，只作历史参考。其中的章节号、路径和部分说法已经过时，一切以新版为准。更早的 v2 见 git 提交 8f22e8f。

> 目标文章：《【生成模型】薛定谔流：玩鸣潮的同时玩原神》（知乎 [ycz](https://www.zhihu.com/people/ycz-19)，2026-04-17）
> 底层论文：[Schrödinger Bridge Flow for Unpaired Data Translation](https://arxiv.org/abs/2409.09347)（De Bortoli et al., NeurIPS 2024）
>
> **v2 说明**：读到原文后，v1 方案有三处根本性错误已修正。见第 0 节。
>
> **v3 说明（2026-11-13）**：明确了最终目标是**实时、无闪烁**的画风转换；训练改在本地 RTX 3090 Ti 上进行；规范分辨率定为 1024×576。决策汇总见下方"v3 决策摘要"，涉及的各节正文已同步修改。
>
> 确定条件：游戏内实机截图 / 白天户外自然景 / 被动截屏采集 / **本地 RTX 3090 Ti 训练（Colab 暂缓）** / 规范分辨率 **1024×576** / 先追画质与内容一致性，再追实时

---

## v3 决策摘要（2026-11-13）

**目标排序**：先在多步 teacher 上拿到好的转换效果和内容一致性 → 再解决时序闪烁 → 最后蒸馏成单步 student 做实时。实时是硬需求，现在做的每个选择都不能堵死这条路。

| 议题 | 决策 | 详见 |
|---|---|---|
| 训练硬件 | 本地 RTX 3090 Ti 24 GB；Colab 和大规模数据暂缓 | §6.4 |
| 规范分辨率 | 1920×1080 → **1024×576**（16:9，32 的倍数），latent 32×18×32 = 576 token；训练与推理同一像素尺度 | §3.2, §5.4 |
| 训练样本 | 预计算**整帧** latent（含水平翻转），训练时在 latent 上**现裁** 16×16 窗口；不预建 patch 库 | §3.2, §4.1 |
| OT 代价 | 在裁出来的区域上算，且**逐位置对齐**：RGB 缩略图 MSE，或 DINOv2 patch token 逐位置余弦距离（不用池化 / CLS） | §4.1 |
| 位置编码 | **2D RoPE**（相对位置），不用绝对坐标，减少画面平移时的不等变 | §4.2 |
| 推理 | **整帧单次前向，不切 tile**；是否混整帧 batch 由"整帧 vs 逐 crop"一致性测试决定 | §4.4 |
| EDM 系数 | 正反向**分别标定** | §2.5 |
| 评估 | 保留按 session 切的 val 集；速度 val loss 只当过拟合警报；主要看固定评估面板（对照图 + KID / 结构距离 / 记忆化等） | §7 |
| 时序 | 先测（平移等变测试、warping error），再按成本从低到高修 | §7.4 |
| 实时 | teacher → 单步 student 蒸馏；DC-AE 实测太慢，student 大概率做在 pixel 空间 | §10 |

**v2 中被推翻或修正的说法**：
- "往返一致性是学到了桥的有力证据"（v2 §7.3）：不成立。正反方向积分的是同一个 ODE 场，连续时间下任何向量场的往返都恒等于原图，误差只反映离散化，适合用来定 NFE
- "帧间连贯性是免费的；闪烁明显就说明 ODE 实现有问题"（v2 §1.1、§4.4、§7.3）：太强。确定性只消除了"每帧重新采样噪声"带来的闪烁；f32 AE 和 32 px token 网格的混叠在 ODE 完全正确时也会造成闪烁
- "正反向共用一套 EDM 系数"（§2.4）：前提是两域 latent 的方差和均值相同，实际不一定，改为分别标定
- DiT-B 参数量（§4.2）：含 adaLN 调制约 130M；约 85M 只是按 token 计算的主体
- 512×512 中心裁的训练方案（§5.4）：改为 1024×576 整帧 latent + 现裁窗口

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

**这带来三个实际好处**：推理确定性、帧间连贯性好得多（作者展示的视频没加 temporal 约束也很平滑）、NFE 可以压得很低。

> v3 修正：确定性只消除了"每帧重新采样噪声"这一类闪烁。f32 AE 与 token 网格的混叠、OT 映射在歧义区域的陡变，仍可能造成闪烁，见 §7.4。

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

> v3 注意：同形的前提是 §2.2 的统计假设，即两域 latent 的方差（和均值）相同。真实数据不一定满足，实现上正反向分别标定，见 §2.5。

### 2.5 工程实现

**不要用解析式硬套高斯假设。** 真实潜空间分布不是高斯，OT 配对后的 ρ 也随数据而变，两域的方差和均值也不一定相同。更稳的做法是**离线实测，正反向各标一张表**：

```python
@torch.no_grad()
def calibrate_coeffs(z0_pool, z1_pool, eps, pair_fn, n_tau=64, n_sample=4096):
    """在 τ 网格上实测三个统计量；正反向各一张表（d=1 正向，d=0 反向），查表 + 线性插值"""
    taus = torch.linspace(1e-3, 1-1e-3, n_tau)
    tables = {}
    for d in (1, 0):
        rows = []
        for tau in taus:
            z0, z1 = pair_fn(sample(z0_pool, n_sample), sample(z1_pool, n_sample))
            t  = tau if d == 1 else 1 - tau           # 网络时间 τ → 物理时间 t
            z  = torch.randn_like(z0)
            xt = (1-t)*z0 + t*z1 + (eps*t*(1-t)).sqrt()*z
            v  = (z1 - xt) / (1 - t) if d == 1 else (z0 - xt) / t

            var_x = xt.var().item()
            var_v = v.var().item()
            cov   = ((xt - xt.mean()) * (v - v.mean())).mean().item()

            c_skip = cov / var_x
            c_out  = max(var_v - cov**2 / var_x, 1e-8) ** 0.5
            c_in   = var_x ** -0.5
            rows.append((c_in, c_skip, c_out))
        tables[d] = torch.tensor(rows)
    return taus, tables
```

跑一次（几分钟）存成查表，训练和推理时按 d 选表、按 τ 插值取值。这样完全不依赖高斯假设，也自动吸收了 OT 配对带来的 ρ 和两域统计量的差异。`pair_fn` 要用和训练完全相同的现裁 + 配对流程（§4.1）。

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

**推荐 `f32c32-sana-1.1`**：512px 输入只有 256 token，DiT 训练极快，且 f32 的重建质量明显好于 f64/f128。游戏画面细节多，不建议上 f64 以上。v3 的规范分辨率 1024×576 编码后是 32×18 = 576 token（整帧推理），训练用 16×16 窗口 = 256 token。

> ⚠️ `in` 和 `mix` 变体的 checkpoint key 相同但 **scaling_factor 不同**，`from_single_file` 无法自动区分（会默认按 `mix` 配）。用 `from_pretrained` 加载 `-diffusers` 版本可避开这个坑。

### 3.2 潜空间预计算（v3：整帧 1024×576）

**潜空间必须离线预计算并缓存**，不要在训练循环里现编码——那会让 AE 成为瓶颈，且浪费 GPU（3090 Ti 上 1024×576 编码一次约 80 ms，见 §3.3）。

v3 起缓存的是**整帧** latent：1920×1080 缩到规范分辨率 1024×576，编码得到 32×18 的 latent 网格。水平翻转作为单独的样本，**翻转后重新编码**（AE 不是严格翻转等变的，不在 latent 上直接翻）。训练时在整帧 latent 上现裁窗口（§4.1），推理时整帧单次前向（§4.4），两边看到的都是"整帧编码出来的 latent"，分布一致，比单独编码 crop 对得更齐。

```python
import torch
from diffusers import AutoencoderDC

ae = AutoencoderDC.from_pretrained(
    "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", torch_dtype=torch.float16
).to("cuda").eval()

sf = ae.config.scaling_factor   # 读配置，不要硬编码（实测 0.41407）

@torch.no_grad()
def encode_batch(imgs):              # imgs: (B,3,576,1024) in [-1,1]，原图与水平翻转各一份
    z = ae.encode(imgs).latent        # (B,32,18,32)
    return (z * sf).to(torch.float16) # 标准化到单位方差后存 fp16
```

编码用 fp16 即可：实测与 fp32 输出相差约 76 dB PSNR，无 NaN（§3.3）。

**同时缓存与 latent 网格对齐的配对特征**（OT 配对要用，见 4.1）：

```
data/30_packed/
  genshin_lat.npy    (2N, 32, 18, 32)  float16   # N 帧 + 翻转；N=5000 ≈ 370 MB
  genshin_thumb.npy  (2N, 3, 36, 64)   uint8     # 1/16 缩略图，每个 latent 格对应 2×2 像素；≈ 69 MB
  genshin_dino.npy   (2N, 64, 18, 32)  float16   # 可选：DINOv2 patch token（448×252 输入 → 32×18 网格），PCA 到 64 维；≈ 740 MB
  genshin_meta.csv   每行：源文件、session、是否翻转
  wuwa_*             同上
```

全部可一次性放进显存，训练时零 I/O。DINO 特征只在 RGB 代价不够用时才需要（§4.1）。

**务必先验证重建质量**：取 50 张图过一遍 `encode → decode`，人眼检查描边、远景植被、水面高光有没有糊掉。如果 f32 重建就已经丢了画风关键特征，后面训什么都没意义。（已在 512 中心裁上验证通过；换到 1024×576 后抽查一次即可。）

### 3.3 编解码耗时（RTX 3090 Ti 实测，batch 1，2026-11-13）

| 分辨率 | fp32 编码 + 解码 | fp16 / bf16 编码 + 解码 |
|---|---|---|
| 512×512 | 157 ms | 91 ms |
| **1024×576** | 345 ms | **≈200 ms**（编码 80 + 解码 120） |
| 1280×704 | 522 ms | 303 ms |
| 1920×1088 | 1211 ms | 720 ms |

encoder 153M、decoder 159M 参数。计算受限：channels_last 没有收益，估计换 TensorRT 最多快 1.5 倍左右。

**结论**：训练不受影响（latent 离线预计算，只在评估时解码）；但 **DC-AE 进不了实时回路**，见 §10。

---

## 4. 训练

### 4.1 minibatch OT 配对（内容一致性的来源）

**关键：在与 latent 网格对齐的 pixel 缩略图（或 DINO 特征）上配对，在 latent 上训练。**

v3 起每一步的流程是：两域各随机取 B 帧整帧 latent → 现裁同尺寸的窗口 → 在**裁出来的区域**上算代价 → 解 OT → 配对。不预建 patch 库：两种做法采到的 patch 分布相同，但预建库每域约 8 GB（每帧 17×3 个窗口，再乘翻转），现裁只是一次切片。

两域样本数相等、权重均匀时，精确 OT 的解就是一个排列（线性指派），用 `scipy.optimize.linear_sum_assignment` 即可，不需要 torchcfm：

```python
import random
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def sample_pairs(lat0, lat1, th0, th1, B, win=16, ot_ratio=0.8, cost="rgb", dino0=None, dino1=None):
    """lat*: (M,32,18,32) 整帧 latent；th*: (M,3,36,64) 1/16 缩略图（每个 latent 格 = 2×2 像素）；
    dino*: (M,64,18,32) 可选的 DINO patch token"""
    i0 = torch.randint(len(lat0), (B,))
    i1 = torch.randint(len(lat1), (B,))
    oy, ox = random.randint(0, 18 - win), random.randint(0, 32 - win)   # 本 batch 共用一个裁窗（分层采样）
    z0 = lat0[i0, :, oy:oy+win, ox:ox+win]
    z1 = lat1[i1, :, oy:oy+win, ox:ox+win]

    k = int(B * ot_ratio)                          # 前 k 对走 OT，其余保持随机配对
    if cost == "rgb":                              # 逐像素对齐的平方误差
        ys, xs = slice(2*oy, 2*(oy+win)), slice(2*ox, 2*(ox+win))
        f0 = th0[i0[:k], :, ys, xs].flatten(1).float()
        f1 = th1[i1[:k], :, ys, xs].flatten(1).float()
        C = torch.cdist(f0, f1).pow(2)
    else:                                          # DINO：逐位置 token 余弦相似度（不池化）
        d0 = F.normalize(dino0[i0[:k], :, oy:oy+win, ox:ox+win].flatten(2).float(), dim=1)
        d1 = F.normalize(dino1[i1[:k], :, oy:oy+win, ox:ox+win].flatten(2).float(), dim=1)
        C = -torch.einsum("adp,bdp->ab", d0, d1) / (win * win)
    _, j = linear_sum_assignment(C.cpu().numpy())
    z1[:k] = z1[:k][torch.as_tensor(j, device=z1.device)]
    return z0, z1
```

**代价必须逐位置对齐。** 布朗桥 $x_t=(1-t)x_0+t\,x_1$ 是逐位置插值的。配上的两块如果构图对不齐（左边的树配到右边的树），网络就得学"把东西挪位置"，结果是内容漂移。RGB 缩略图 MSE 天然对齐；DINO 代价要按 patch token 逐位置比，**不要用平均池化或 CLS**，那只比"画面里有什么"，不比构图。

**候选集怎么组：**
- **默认**：随机取 B 帧，本 batch 共用一个裁窗。这相当于分层采样，候选的构图更整齐，OT 更容易找到好伙伴。位置编码用 RoPE（§4.2），不依赖裁窗位置，所以每张各自随机裁也可以
- **可选：配对池大于训练 batch**。例如在 512 张缩略图上解一次 OT，分 4 步用完。OT 近似更好，训练 batch 不变；配合 `ot_ratio` 用
- **以后再考虑：检索式组 batch**。在全库上用特征做 kNN，把相似的 patch 凑进同一个 batch 再解 OT。更接近全局 OT，但也更难学、多样性下降，和调高 OT 比例是同一个方向。pilot 里配对质量明显不够时再上

**可调旋钮**（前两个来自原文第七、八节）：

| 旋钮 | 作用 |
|---|---|
| **代价函数** | RGB MSE → 倾向学颜色相近的转移；**DINO 逐位置相似度** → 倾向保留高层语义做转移 |
| **OT / 随机配对比例** | 控制转移语义的粗细 |
| 裁窗尺寸 | 越小候选越同质、配对越近，但会丢掉天空渐变、大气透视等全局画风线索；从 16×16 起步 |

> 前两个旋钮直接回应了"会不会只学到调色"的担忧。如果 RGB 代价下输出只是换色，**切 DINO 代价**——它强制按语义配对，颜色相似不再是配对依据。这比我 v1 方案里设计的一堆诊断指标更直接：它是从源头解决，不是事后检测。
>
> DINO 用 `facebook/dinov2-small`（或带 registers 的版本）。整帧以 448×252 输入，patch 网格正好 32×18，和 latent 网格逐格对齐；离线算一次存盘（§3.2）。

`ot_ratio` 建议从 0.8 起步。全 OT（1.0）配对过硬会让多样性下降。

### 4.2 网络：DiT + AdaLN + 2D RoPE

作者第八节特别指出：用 DiT 的 AdaLN 注入 t 和方向 d 时，**正反速度共用网络主体，推理做 `(v→−v←)/2` 相减时内部偏差会大量抵消**，因此在 ε 很低时能急速收敛。这是选 DiT 而非 UNet 的实质理由，不是跟风。

f32c32 的配置（DiT-B，约 130M 参数，其中 adaLN 调制约 42M，按 token 计算的主体约 85M）：

```python
DiTConfig(
    patch_size=1, in_channels=32,                    # token 网格可变：训练 16×16 窗口 = 256 token，推理整帧 32×18 = 576 token
    hidden_size=768, depth=12, num_heads=12,
    pos="rope2d",                                    # 2D 轴向 RoPE（相对位置），不用绝对 sincos pos_embed
    cond=["tau", "direction"],                       # 都走 AdaLN modulation
)
```

方向 d 用 `nn.Embedding(2, hidden_size)`，与 τ 的 embedding **相加**后一起进 AdaLN。pilot 用 DiT-S（hidden 384、depth 12、heads 6，约 33M）。

**位置编码用 2D RoPE，不用绝对坐标。** 用绝对位置时，同一块内容平移到屏幕另一处会拿到不同的位置嵌入，输出跟着变，这是时序上"等变性项"的来源之一（§7.4）。RoPE 下 attention 只依赖相对位移，内容按整 token 平移时输出也跟着平移（边界附近除外）。代价是训练窗口内的相对距离最大只有 15，整帧推理要用到 31，这也是要做 §4.4 一致性测试、必要时混整帧训练的原因。

基于 [facebookresearch/DiT](https://github.com/facebookresearch/DiT) 的 `models.py` 改：`y`（类别标签）那一路换成 direction embedding；去掉 `pos_embed`，在每层 attention 的 q/k 上加 2D RoPE；`unpatchify` 支持可变网格。

### 4.3 训练循环

```python
EPS = 0.05          # 低 ε，见 4.5

def train_step(net, ema, opt, data, coeff_tables, B=128):
    z0, z1 = sample_pairs(**data, B=B, win=16, ot_ratio=0.8)   # §4.1：现裁 + 对齐代价 + OT

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

    # EDM 重参数化：按 d 选表、按 tau 插值（§2.5）
    c_in, c_skip, c_out = lookup(coeff_tables, tau, d)
    target = (v - c_skip*xt) / c_out
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = net(c_in * xt, tau, d)

    loss = F.mse_loss(pred.float(), target)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step(); ema.update(net)
    return loss.item()
```

**训练循环里没有任何 SDE 求解。** 这就是成本能压到单卡一天的原因。3090 Ti 上 DiT-B 尺寸、256 token × batch 128 实测约 0.40 s/步（§6.4）。

整帧样本（32×18，batch 约 64）可以作为另一个尺寸桶混进来，同一个 batch 内尺寸一致即可；要不要混，由 §4.4 的一致性测试决定。

### 4.4 推理（ODE）

```python
@torch.no_grad()
def translate(net_ema, z_src, n_steps=32, direction="0to1", coeff_tables=None):
    """z_src: 整帧 latent (B,32,18,32)；整帧单次前向，不切 tile"""
    x = z_src.clone()
    B = x.shape[0]
    dt = 1.0 / n_steps
    sgn = 1.0 if direction == "0to1" else -1.0
    d = torch.cat([torch.ones(B), torch.zeros(B)]).to(x.device)

    for i in range(n_steps):
        t = i*dt if sgn > 0 else 1 - i*dt
        tt = torch.full((B,), t, device=x.device)
        tau = torch.cat([tt, 1 - tt])                  # 正向 τ=t，反向 τ=1-t
        v = denoise(net_ema, torch.cat([x, x]), tau, d, coeff_tables)   # 两个场拼成一次前向
        v_f, v_b = v[:B], v[B:]
        x = x + sgn * 0.5 * (v_f - v_b) * dt          # 概率流 ODE
    return x
```

每步两个速度场，拼成一次前向（3090 Ti 上 DiT-B 整帧约 6 ms/步）。`n_steps=32` 起步，可往下压到 16 看质量损失；评估时记录 NFE–质量曲线（§7.3），为实时化做准备。

**整帧单次前向，不切 tile。** 独立 tile 各自做全局 attention，重叠区看到的上下文不同，低频的色调和曝光会出现块状不一致；tile 边界又固定在屏幕上，内容移过边界时上下文突变，视频里会在固定位置闪烁。以后分辨率更高、确实必须切 tile 时，也不要拼最终结果，而是在**每个 ODE 步**对重叠区的速度加权平均（MultiDiffusion 的思路），让轨迹联合演化。

**整帧 vs 逐 crop 一致性测试**（决定要不要混整帧训练）：同一张图整帧推理一次，再切成 16×16 窗口逐块推理，比较窗口内部区域两者的差别。
- 几乎一致：模型对上下文大小不敏感，不用混
- 整帧版色调明显偏移：把整帧 batch 作为尺寸桶混进训练（例如每 4 步 1 步，batch 约 64，可配合更大的配对池），或者最后用整帧（或整帧与 crop 各半）短训 1–2 万步收尾。收尾阶段配对变粗，内容一致性可能小幅回退，要用评估面板盯着
- pilot 只用 crop；main 之前做这个测试

> **帧间连贯性**：ODE 确定性消除了"每帧重新采样噪声"带来的闪烁，作者的视频 demo 没有加 temporal 约束也很平滑。但这不是全部：f32 AE 与 token 网格的混叠仍可能造成闪烁，见 §7.4。

### 4.5 超参表

| 超参 | 值 | 来源 |
|---|---|---|
| **ε** | **0.05**（低 ε） | ✅ 原文第八节：低 ε 急速收敛；高 ε 退化成两个 VE Diffusion，收敛时间数倍以上，4090 扛不住 |
| OT 配对比例 | 0.8 | ⚙️ 我校准（原文只说可调） |
| OT 代价函数 | RGB 起步，必要时切 DINO；都逐位置对齐 | ✅ 原文第七节；对齐方式 🆕 v3 |
| 配对空间 | 与 latent 网格对齐的 1/16 缩略图（16×16 窗口 → 32×32 像素）/ DINOv2 patch token | ✅ 原文第七节；对齐方式 🆕 v3 |
| 训练空间 | DC-AE f32c32 latent | ✅ 原文第七节 |
| 规范分辨率 | 1024×576 → latent 32×18 | 🆕 v3 |
| 训练窗口 | 整帧 latent 上现裁 16×16（256 token）；是否混整帧桶见 §4.4 | 🆕 v3 |
| 网络 | DiT-B + AdaLN（~130M）；pilot 用 DiT-S（~33M） | ✅ 原文第八节；尺寸 ⚙️ |
| 位置编码 | 2D RoPE（相对位置） | 🆕 v3 |
| 重参数化 | EDM 形式，系数见第 2 节；正反向分别标定 | ✅ 原文第六节（系数由我推导补全）；分别标定 🆕 v3 |
| 推理 | ODE，`(v→−v←)/2`；整帧单次前向，不切 tile | ✅ 原文第五节；整帧 🆕 v3 |
| 优化器 | AdamW，lr 1e-4，wd 0.0 | ⚙️ 我校准（DiT 常规） |
| warmup | 2000 步线性 | ⚙️ 我校准 |
| batch size | 128（bf16，16×16 窗口；3090 Ti 实测 0.40 s/步、峰值 13 GiB）；整帧桶约 64 | ⚙️ 我校准；实测 🆕 v3 |
| 训练步数 | pilot 2–3 万；main 150k（3090 Ti 约 17–20 h） | ⚙️ 我校准 |
| EMA decay | 0.9999 | ⚙️ 我校准 |
| τ clamp | [1e-3, 1−1e-3] | ✅ 奇点要求 |
| 推理 NFE | 32（正反两个场拼成一次前向）；评估时记录 NFE–质量曲线 | ⚙️ 我校准 |

> **ε 是最重要的旋钮**，原文说它控制"转移的生硬程度"。0.05 是保守起点，可在 0.01–0.2 间扫。ε 调大画风改动更剧烈但收敛变慢。v3 补充：ε 也影响时序稳定性，ε 越大映射通常越平滑，歧义区域越不容易来回翻（§7.4）。

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

### 5.4 缩放与打包（v3）

- 整帧 1920×1080 → resize **1024×576**（Lanczos / area），不再中心裁正方形；训练和推理是同一像素尺度
- 水平翻转两域都开：翻转后重新编码，作为单独样本存
- 过 DC-AE 得整帧 latent（32×18），同时存与 latent 网格对齐的 1/16 缩略图，以及可选的 DINO patch token（见 3.2）
- 在线增强 = 训练时在整帧 latent 上随机裁窗（§4.1），不需要再从原图裁
- **按采集 session 切分 train/val**，不要随机切（同 session 帧高度相关会泄漏）。s05（混合地貌会话）作 val；s00 试采可并入 val 或丢弃

### 5.5 公开数据集情况

搜索结论：**没有可直接用的。**

- 原神侧只有 [svjack/video-dataset-genshin-impact-landscape-organized](https://huggingface.co/datasets/svjack/video-dataset-genshin-impact-landscape-organized)（60 条 / 16 MB）和 [ep-landscape 版](https://huggingface.co/datasets/svjack/video-dataset-genshin-impact-ep-landscape-organized)（93 条 / 45 MB），且是**官方 EP 宣传动画过场**不是实机画面，授权不明
- 其他 Genshin 数据集（[animelover](https://huggingface.co/datasets/animelover/genshin-impact-images)、[Fazzie/Teyvat](https://huggingface.co/datasets/Fazzie/Teyvat)）全是角色同人插画
- **鸣潮侧 HuggingFace 上一张场景图都没有**，全是语音/TTS（[wwTTS](https://huggingface.co/datasets/Genius-Society/wwTTS)、[AI-Hobbyist](https://github.com/AI-Hobbyist/WutheringWaves_Datasets)）和 datamine 资源（[Arikatsu](https://github.com/Arikatsu/WutheringWaves_Data)、[anyliew](https://github.com/anyliew/ww_resources)）

自己采集反而更好：同机器同设置、无编码伪影，域间差异干净地只剩画风。

### 5.6 版权与边界

两款游戏画面均为著作权作品。数据仅供个人研究学习：不公开再分发采集数据集，不用产出模型提供商业服务。

采集手段限于**被动屏幕捕获**（§5.2 的安全边界）：不读写游戏内存、不注入进程、不模拟输入、不触碰反作弊相关的任何东西。

### 5.7 时序测试片段（v3 新增）

§7.4 的 warping error 需要**连续帧**，而正式采集是 1.5 fps 且带去重，用不了。每个游戏另录 2–3 段 20–30 秒、30 fps 的连续画面：不去重、不过滤，覆盖匀速步行、慢速转镜头、植被和水面，直接存 1024×576。`capture.py` 需要加一个 clip 模式（只是换一种保存策略，§5.2 的安全边界不变）。

---

## 6. 训练工程（v3：本地 3090 Ti 优先，Colab 暂缓）

> **v3**：先在本地 RTX 3090 Ti 上训练，见 §6.4。6.1 的 Colab 预算留作以后上云参考；6.2 的断点续训、6.3 的目录结构本地同样适用（Drive 换成本地目录）。

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

### 6.4 本地 RTX 3090 Ti（v3 当前方案）

**环境**（2026-11-13 搭好）：仓库根目录下的 `.venv`，Python 3.12.11，torch 2.11.0+cu128（3090 Ti 是 sm_86，和采集机用同一个 cu128 轮子），diffusers 0.43、transformers 5.23、pot、einops、lpips、tensorboard；DC-AE 权重已在 HF 缓存。训练数据需从采集机拷到 `data/10_frames/`。

**实测**（bf16）：

| 项目 | 结果 |
|---|---|
| bf16 矩阵乘 | 76 TFLOPS（峰值约 80） |
| DiT-B 尺寸训练一步（256 token × batch 128） | 0.40 s，峰值显存 13 GiB |
| 150k 步 | 约 17 h，加上 OT 和数据开销约 19–20 h |

原作者"4090 一天"的配置，在这里大约要 1–2 天，所以不能每个超参都跑完整训练，分两档：
- **Pilot**：DiT-S，16×16 窗口，2–3 万步，每次估计 1–2 小时。扫描顺序：先在 RGB 代价下扫 ε ∈ {0.02, 0.05, 0.1, 0.2}，再比 RGB vs DINO，最后调 OT 比例
- **Main**：DiT-B，150k 步以上；按 §4.4 的一致性测试决定是否混整帧或用整帧收尾

**工程**：
- bf16 autocast + TF32
- 打包好的数据整个放进显存，不用 DataLoader（Windows 上多进程开销大）
- Windows 上 torch.compile 的 GPU 后端依赖 Triton，不一定开箱可用，先不依赖它
- 原子 checkpoint + 启动自动 resume（§6.2）；Windows 电源设置关掉睡眠
- 日志用 TensorBoard（曲线 + 评估面板对照图）

---

## 7. 评估

作者坦承原文没做定量比较。但你该做一些，至少用于判断训练是否走偏。

### 7.0 val 集的用途（v3）

**要 val 集，但不拿它看速度 MSE。** 速度 loss 在这个任务里基本没有指示作用：
- **不可约部分占大头**：目标里有布朗噪声，配对也是随机的（同一张图在不同 batch 里配到的伙伴不同），曲线很早就平了，模型好坏只体现在很小的可约部分上
- **跨配置不可比**：ε、OT 比例、代价函数会直接改变目标本身
- **看不出内容一致性**：内容一致性来自配对（耦合），loss 只衡量"这个耦合拟合得好不好"；随机配对也能拟合得很好

速度 loss 照常记录；val loss 用固定 τ 网格、固定噪声、固定配对计算，**只拿 train/val 差距当过拟合警报**。

val 集（按 session 切，§5.4）真正的用处：
- **查记忆化**：150k 步 × batch 128、每域 3–5k 张，每张图要被看 4000–6000 次。记忆化的表现是输出"借用"训练集目标域某帧的内容（凭空多出一块石头），这正是内容一致性的失败，只有没见过的输入才测得出来。实时推理时每一帧都是没见过的
- **固定评估面板**（7.1–7.3）

### 7.1 主要看图

固定的 val 帧（整帧 1024×576），用 EMA 模型、固定 NFE 跑到 ODE 终点，出对照网格（`源图 | 转换后`），两个方向各一张。这是最有效的诊断手段，作者展示的也是这种对照图。
- **快速面板**：每 2–5k 步，每方向 32 张，出图 + LPIPS / 结构距离
- **完整面板**：每 2 万步，每方向 256 张，算 7.3 的全部指标

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

### 7.3 评估面板指标（v3）

| 指标 | 衡量什么 | 怎么读 |
|---|---|---|
| **KID**（DINOv2 特征，对 held-out 目标域真实帧） | 像不像目标域（第一要素） | 越低越好；样本只有几百张时用 KID，不用 FID |
| **DINO self-similarity 距离**（输入 vs 输出） | 结构 / 内容是否保留（第二要素），对画风不敏感 | 越低越好；Splice / PnP 系列用它衡量结构 |
| LPIPS(输入, 输出) | 改动幅度 | 本身无好坏，和上两项一起看 |
| tone-collapse ρ（7.2） | 是不是只调了色 | 见 7.2 |
| 输出到训练集目标域的最近邻 DINO 相似度 | 记忆化 | 和"held-out 真实目标帧到训练集"的同一统计量比，明显更高就是在背图 |
| 4 步 vs 32 步输出差 | 轨迹直不直 | 用来定 NFE，为实时化（§10）做准备 |
| 整帧 vs 逐 crop 输出差（窗口内部区域） | 对上下文大小是否敏感 | 决定要不要混整帧训练（§4.4） |

**读法：**
- KID 和结构距离是一对 trade-off，ε 和 OT 比例就是在这条前沿上移动，没有单一最优数字。同一个 run 里两者应该一起变好、然后进入平台期；**KID 还在降、结构距离却开始升**，说明开始幻觉或背图，停训或回退 checkpoint
- 配两个 baseline，数字才有意义：一是"不转换"（输入直接和目标域比 KID），二是 Reinhard 全局调色。模型比全局调色好不了多少，就只是在调色
- **往返一致性**（`A → B → A`）**不能**证明学到了桥：正反两个方向积分的是同一个 ODE 场，连续时间下任何向量场的往返都恒等于原图，误差只反映离散化和轨迹弯曲。可以和"4 步 vs 32 步"一起用来定 NFE
- FID 只在 val 集够大（≥ 500 张）时补充计算，且只横向比 checkpoint，不和论文数字比

### 7.4 时序一致性（v3）

确定性 ODE 消除了最大的闪烁来源（每帧重新采样噪声），但确定性本身不保证连贯。帧间误差可以拆成两项（$\mathcal{W}$ 为按光流 warp）：

$$
T(x_{t+1}) - \mathcal{W}[T(x_t)]
= \underbrace{T(x_{t+1}) - T(\mathcal{W}x_t)}_{\text{Lipschitz 项}}
+ \underbrace{T(\mathcal{W}x_t) - \mathcal{W}[T(x_t)]}_{\text{等变性项}}
$$

- **等变性项很可能是主因**：DC-AE 是 32 倍下采样，token 网格也是 32 px。镜头移动不到 32 px 时，内容落在 token 里的相位变了，输出不等于平移后的输出。表现是纹理游动，植被最明显（类似 StyleGAN3 要解决的 texture sticking）
- **Lipschitz 项**：输入本身的帧间变化（TAA 抖动、植被动画、遮挡）被映射放大。ε 越低映射越接近 OT，而 OT 映射在多模态分布下可以不连续，在"画成石头还是草"这类歧义区域容易翻转；ε 调大映射更平滑，这是 ε 的另一层 trade-off

**先测：**
- **平移等变测试**（不需要视频）：把一帧平移 k = 1…31 px，转换后再移回来，和不平移的结果比，画出误差随 k 的曲线。只做 AE 重建的那条曲线就是闪烁的下限
- **warping error**：在连续测试片段（§5.7）上用 RAFT 光流加遮挡 mask，对比输入、AE 重建、转换输出三者

**再修**（成本从低到高）：
1. 训练时加亚 token 相位增强（多编几个像素偏移的 latent）
2. 调 ε 和配对方式
3. **蒸馏 student 时加时序 / 扰动一致性损失**：主力手段，和实时化在同一阶段（§10）；卷积 student 本身更接近平移等变
4. 兜底：推理时做光流引导的时域融合

> 修正 v2：v2 写"闪烁明显就说明 ODE 实现有问题"太强。ODE 完全正确也会因混叠而闪烁，先看等变测试再下结论。符号或 τ 传错一般在单帧对照图上就能看出来（输出跑偏或糊掉）。

---

## 8. 实现顺序、依赖与风险

### 8.1 建议顺序（v3）

**已完成**：采集脚本与数据采集（用户确认采集链路没问题）；DC-AE 重建闸门（512 中心裁，用户确认通过）；训练机 3090 Ti 的环境与测速（§3.3、§6.4）。

| 阶段 | 内容 | 产出 |
|---|---|---|
| 0 | 数据拷到训练机；平移等变测试（AE 单独）；1024×576 下抽查 DC-AE 重建 | 闪烁下限；确认规范分辨率下重建仍合格 |
| 1 | 离线过滤 + 配平（§5.3）；整帧 latent 打包（含翻转）+ 缩略图（+ 可选 DINO）；正反向系数标定；评估面板 | 数据与评估就绪 |
| 2 | DiT 改造（direction + 2D RoPE + 可变网格）；冒烟测试 2k 步；pilot 扫 ε / 代价 / OT 比例 | 选定配置 |
| 3 | main 训练 teacher；整帧 vs 逐 crop 一致性测试，决定是否混整帧 | 高质量 teacher |
| 4 | 录时序测试片段（§5.7），做 teacher 的时序评估（§7.4） | 闪烁报告 |
| 5 | 蒸馏单步 student（带时序正则）→ 实时 demo（§10） | 实时版 |

### 8.2 第三方依赖清单

**核心（缺了跑不起来）**

| 库 | 用途 | 节 |
|---|---|---|
| `torch` | 基础（cu128 轮子：采集机 RTX 5070 是 sm_120，**必须 cu128+**；训练机 3090 Ti 是 sm_86，同一个轮子） | 全文 |
| `diffusers` | `AutoencoderDC` —— DC-AE 编解码（需较新版本） | §3.2 |
| `numpy` | `.npy` 打包、统计、OT 索引 | §3.2, §4.1 |
| `Pillow` | 图像读写 | §5 |
| `scipy` / `pot` | OT 求解：两域等量、均匀权重时精确 OT 就是线性指派（`scipy.optimize.linear_sum_assignment`）；`pot` 留作非等量 / Sinkhorn | §4.1 |

> v3 不再依赖 `torchcfm`：两域样本数相等时 OT plan 就是一个排列，一次线性指派即可（§4.1）。

**数据采集与过滤**

| 库 | 用途 |
|---|---|
| `mss` | 快速截屏（比 `PIL.ImageGrab` 快一个量级） |
| `pywin32` | 窗口定位、焦点检测 |
| `imagehash` | pHash 去重 |
| `opencv-python` | Laplacian 方差、边缘密度（也可用 numpy 手写） |
| `keyboard` / `pynput` | 全局热键，**只读键盘不发按键**（可选，见 §5.2.1） |
| `ffmpeg` | 仅 §5.2.3 备选路线需要（命令行工具，非 Python 包） |

**评估与日志**

| 库 | 用途 |
|---|---|
| `lpips` | 改动幅度、色调退化检测 ρ |
| `transformers` | DINOv2：KID 特征、self-similarity 结构距离、记忆化最近邻、DINO 代价函数（KID 自己算，不需要 clean-fid） |
| `torchvision` | RAFT 光流（§7.4 warping error） |
| `tensorboard` | 曲线 + 评估面板对照图 |
| `opencv-python` | Reinhard 色彩迁移的 Lab 转换（不需要 scikit-image） |

**可选**

| 库 | 用途 | 何时需要 |
|---|---|---|
| `einops` | DiT 的张量重排 | 跟 DiT 代码走（已装） |
| `open_clip` / `transformers` | CLIP zero-shot 场景过滤 | §5.3 可选步骤 |
| `clean-fid` | FID | val 集 ≥ 500 张、想补 FID 时 |

**非 pip 包**：[facebookresearch/DiT](https://github.com/facebookresearch/DiT) 不发包。v3 要改的地方变多了（direction embedding、2D RoPE、可变网格，见 §4.2），可以 clone 后改 `models.py`，也可以参照它自写一个精简版。

### 8.3 风险清单

| 风险 | 概率 | 应对 |
|---|---|---|
| EDM 系数实现错误 | **高** | 第 2.3 节的 τ=0/1 单位方差自检；用实测标定而非解析式；正反向分别标定 |
| ODE 符号 / 时间传错 → 输出跑偏或糊 | **中高** | 单帧对照网格；单步手工验算 |
| AE 与 token 网格混叠 → 闪烁 | **中高** | §7.4 平移等变测试；相位增强；student 时序正则 |
| 小数据记忆化（每张图被看几千次） | 中 | §7.0 val 集 + 最近邻检查；现裁窗口扩数据；早停 |
| 只学到调色 | 中 | 切 DINO 代价（7.2） |
| OT 配对在小 batch 上质量差 | 中 | 强约束场景（5.1）；现裁窗口；配对池大于训练 batch（§4.1） |
| 整帧推理与窗口训练不一致 | 中 | §4.4 一致性测试；混整帧桶或整帧收尾 |
| ε 选大了收敛不了 | 中 | 从 0.05 起，别一上来就调大 |
| DC-AE 重建丢画风特征 | 低（512 中心裁闸门已过） | 1024×576 下抽查（阶段 0） |
| DC-AE 太慢，进不了实时回路 | **已确认** | 实时版绕开 DC-AE，student 做在 pixel 空间（§10） |
| 训练中断（Windows 睡眠 / 更新重启） | 中 | 关睡眠；6.2 原子 checkpoint + 自动 resume |
| 在线过滤阈值偏松 / 偏紧 | 中 | 阈值可配 + 统计落盘；§5.3 离线复核 |
| 独占全屏导致截屏拿到黑屏 | 已解决 | 采集已验证可用 |
| 全局热键需要提权 | 低 | 改用窗口焦点检测自动暂停（§5.2.1） |

---

## 9. 待确认

- 采集相关的三项（版本与地区、无边框窗口、热键）：已在实际采集中解决，用户确认采集链路没问题
- 时序测试片段的录制（§5.7）：需要给 `capture.py` 加 clip 模式
- 是否需要混整帧训练：由 §4.4 的一致性测试决定

**已确认的环境**：
- **采集机**：RTX 5070 / 12 GB（驱动 621.01）。Python Install Manager 已装 3.12.10（系统默认是 3.15.0，PyTorch 无 cp315 轮子，须用 3.12）；torch 必须装 cu128 或更新的构建（Blackwell / sm_120，cu121 轮子跑不了）；AppLocker 封 Anaconda 签名的 exe → 不用 Miniconda，用 venv / Miniforge / uv
- **训练机**：RTX 3090 Ti / 24 GB（驱动 610.88），仓库根目录 `.venv`（Python 3.12.11，torch 2.11.0+cu128），见 §6.4

---

## 10. 实时化路线（v3 新增）

最终目标是实时、无闪烁的画风转换，理想情况是单步。现阶段先把多步 teacher 的画质和内容一致性做好，但每个选择都不能堵死这条路。

### 10.1 路线：teacher → student

把多步薛定谔流当作"慢而准"的 teacher，用它生成（输入，输出）对，再蒸馏出一个单步 student。
- **好蒸馏**：teacher 的 ODE 是确定性的，每个输入只对应唯一的目标，蒸馏就是普通的配对回归，不会因为对多个目标取平均而变糊
- **数据不受限**：student 只需要源域帧（任意多的实机画面，包括连续帧），不需要目标域数据
- **时序正则放在这一阶段**：对连续帧或合成平移加一致性损失（§7.4 第 3 条）

### 10.2 实测：瓶颈在 AE

| 组件（3090 Ti，batch 1，1024×576） | 耗时 |
|---|---|
| DC-AE 编码 + 解码（fp16 / bf16） | ≈200 ms（80 + 120） |
| DiT-B 一个速度场（576 token，bf16） | 3.6 ms |
| DiT-B 正反两个场拼成一次前向 | ≈6 ms |

光 DC-AE 往返一次就把帧率压到 5 fps 以下，而且是计算受限，优化空间有限（§3.3）。所以**实时版必须绕开 DC-AE**：student 做在 pixel 空间（小卷积网络，直接帧到帧），或者另外蒸馏一个小 AE。teacher 训练不受影响。

### 10.3 现在就要做的

1. 规范分辨率已定为 1024×576，训练尺度跟着它走（§5.4）
2. teacher 必须在**整帧单次前向**下表现好，因为它要给 student 打标签（§4.4）
3. 评估时记录 NFE–质量曲线（§7.3）
4. 推理时 v→ 与 v← 拼成一次前向（§4.4）

---

## 参考

1. [Schrödinger Bridge Flow for Unpaired Data Translation](https://arxiv.org/abs/2409.09347) — 薛定谔桥流
2. [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)
3. [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/pdf/2302.00482) — OT 配对
4. [Deep Compression Autoencoder](https://hanlab.mit.edu/projects/dc-ae) — DC-AE
5. [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) — EDM 重参数化
6. [TorchCFM](https://github.com/atong01/conditional-flow-matching) — `OTPlanSampler`
7. [facebookresearch/DiT](https://github.com/facebookresearch/DiT) — DiT 骨架
8. [MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation](https://arxiv.org/abs/2302.08113) — 重叠 tile 每步融合（§4.4）
9. [Splicing ViT Features for Semantic Appearance Transfer](https://arxiv.org/abs/2201.00424) — DINO self-similarity 结构距离（§7.3）
10. [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401) — KID（§7.3）
11. [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — 配对代价与评估特征（§4.1、§7.3）
12. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE（§4.2）
13. [Learning Blind Video Temporal Consistency](https://arxiv.org/abs/1808.00449) — warping error（§7.4）
14. [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039) — 光流（§7.4）
15. [Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — 混叠与 texture sticking（§7.4）
