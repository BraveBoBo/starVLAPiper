# 无限记忆 VLA 设计:moment tokens + fla 递归记忆 + QFormer 读出

> 归档日期:2026-07-04。路线两阶段:Phase 1 在本仓库(QwenPI_v3)落地 moment tokens;
> Phase 2 参照本设计把窗口聚合替换为 fla 无限递归 + QFormer 读出。
> 完整改造目标仓库:`myungkyuKoo/HAMLET-Isaac-GR00T`(本地 `~/project/smooth/third_party/HAMLET-Isaac-GR00T`,HEAD 63b4055)。

## 架构(已定)

```
每帧: VLM(冻结/低lr) + n_q=4 个可学习 moment tokens(nn.Parameter 拼 inputs_embeds 尾部,非特殊 token)
      → post-LLM hidden 尾部 n_q 行 = "此刻摘要"
跨帧: moment tokens 直接写入 fla GatedDeltaNet 递归记忆(O(1) 状态,历史无上限)
读出: QFormerReadout —— m(=n_q=4) 个可学习 query 对当前步 fla 输出 cross-attn
注入: 读出结果替换当前帧 KV 尾部(cross_attn)/ adaln(与 HAMLET 相同)
```

组件顺序为用户指定:**fla 输出的结果使用 QFormer**(读出器,非写入前压缩)。
块内双向语义由读出端恢复(fla 全系严格因果);QFormer 的 KV 只取当前步 fla 输出 → 训练/推理零分布差。

## 关键实现决策

| 决策 | 内容 |
|---|---|
| 直接替换 | `MemoryTransformer` → `RecurrentMemory`,不设 window/fla 开关;属性名沿用 `memory_transformer`(setup.py 容忍串零改动);基线=git 上游 |
| fla 集成 | **ops 路径**:`chunk_gated_delta_rule`(训练,K 帧并行)/`fused_recurrent_gated_delta_rule`(推理单步);state=裸 fp32 张量 `(B, layers, H, Dk, Dv)`,None/零=reset,detach=TBPTT;不用 fla Cache |
| mixer 维度 | dim 2048、H=4、Dk=128、Dv=256(expand_v=2)、2 层;状态 1.0 MiB/样本;g/beta kernel 外 fp32;`use_qk_l2norm_in_kernel=True`;short-conv 关 |
| QFormer | 复用本仓库 `starVLA/model/modules/projector/QFormer.py` 的 `CrossAttentionBlock`(其 mask 路径语义反+形状错,弃用 mask);m=n_q=4 → 注入零 mask 改动 |
| 训练 | Phase A:现有 shuffled K 帧窗口 dataloader(state=None/样本,chunk kernel,K 可升 8/16——线性);Phase B(episode 连续 TBPTT sampler)后续;padding clamp 复制帧接受 |
| 环境 | 一律 starVLA conda env(fla 0.5.2 已装,transformers 5.3.0);与 HAMLET repo pins 漂移接受,动工前 Step-0 验证 |

## HAMLET repo 改造点(Phase 2 参照)

- 新文件:`gr00t/model/modules/recurrent_memory.py`(`_GatedDeltaMixer` + `RecurrentMemory`,契约 `(B, K*n_q, d)` in/out + 取尾部)、`gr00t/model/modules/qformer_readout.py`(`QFormerReadout`)。
- 手术位:`gr00t_n1d6.py`(:114-133 构造、moment 分支 :299-388 删 K_target 守卫 + FIFO 换递归态、vision 分支同理、:393 reset)、`setup.py`(:108 容忍串加 qformer_readout;:131 重初始化改走 `reset_parameters()`——**HF from_pretrained 对 missing key 填 torch.empty 垃圾,A_log/dt_bias/query_tokens 非 Linear 会成 NaN 源**)、`gr00t_policy.py`(session 状态 `_memory_cache`→`_memory_state`,fp32 不 cast)。
- 核心验证不变量:整段 chunk 前向 ≡ 逐步 fused_recurrent 带态前向(bf16 rtol 1e-2);fp32 naive_* 参考实现可做 CPU 单测。
- 无 `advance_memory`(grep 证实);只需保留 `prime_only` + 逐行 `reset_memory`。

## Phase 1(本仓库,已批准):moment tokens

**不用特殊 token**(tokenizer+resize embedding 动 VLM 权重形状、冻结时学不动、有 init 写回 bug)。
采用 HAMLET 方式:独立 `nn.Parameter` 拼 `inputs_embeds` 尾部——移植 `smooth/starVLA/starVLA/model/modules/vlm/hamlet_qwen3.py`(transformers 5.3.0 已验证;复刻 HF Qwen3VLModel vision splice + 3D M-RoPE,moment 位置从每样本真实尾延续),适配 PI_v3 layer-wise(`output_hidden_states=True` 返回各层);本仓库落地为 `starVLA/model/modules/vlm/elephant_qwen3.py`(类 `ElephantQwen3Interface`)。
框架侧 `Elephant(原 QwenPI_v3_MomentMemory)` 保留:`BlockCausalMomentMemory` 窗口聚合(Phase 2 换 fla)、K 帧窗口展开、滚动 cache + 逐行 reset、`enabled=False` no-op。

**同族佐证**:HAMLET(ICLR 2026, arXiv:2510.00695);CronusVLA(arXiv:2506.19816)——motion features + FIFO 特征缓存,每帧只过一次 VLM,历史在 post-LLM 特征层聚合。

## 时序段训练(Temporal-Segment + 每步监督)—— 定稿 2026-07-05,已归档待实现

**动机**:滑动窗口训练每帧每 epoch 过 VLM ≈K 次(IO 同);段训练把"每监督信号的 VLM 前向"从 K 降到 1。

**定稿设计**(完整版见批准计划;要点):
1. **段 = stride 网格**:anchor=段首,video `delta_indices=[i*16 for i in range(T)]`(正向);S=16 **必须**等于推理 action-chunk 执行间隔(train/infer 时钟对齐——文献空白,一等设计点);
2. **动作无缝铺装**:S==horizon=16 → `action_indices=range(T*16)` reshape `(T,16,7)`,段内 T 个 chunk 无重叠无空洞;
3. **每步监督**(收益唯一来源):fla 整段 chunk 扫保留全部步输出 → `(B*T, n_q, d)` 批量过 QFormerReadout → 替换**所有** DiT 层尾部 → B*T 行独立出 flow-matching loss;只监督最后一步则与滑动窗口零差异;
4. **burn-in**(R2D2):段前 `burn_in_steps`(默认 T/4)只暖状态不计 loss,治段中零状态失真;
5. **step_valid_mask**:dataset 子类自算(管线的 padding_positions 算完即弃),episode 末尾 clamp 步不进 loss;`LayerwiseFM_ActionHeader` 加可选 `loss_mask=None`(默认行为不变);
6. **训练配方**:有效 batch 按独立段数计(√B LR 缩放);T∈{4,8,16} 扫描;**记忆置零消融必做**(防每步监督下模型忽略记忆作弊)。

**文献支撑**:范式 = Decision Transformer/Decision Mamba/LRAM(xLSTM,chunk 训 + 递归推);B×T 展平 forward = CronusVLA 公开代码同构(但其只监督最后一步,1:M);"可训练线性注意力递归 + 每步 flow-matching"组合无公开先例。fla `cu_seqlens`(batch=1 打包自动边界重置)为多段打包备选;Dreamer 携带态 TBPTT 为 Phase B。

**本仓库缺口**(审计结论):`sequential_step_sampling` 为死字段勿复用;正向 delta/collate/DiT 行独立均已支持;需新增:dataset 段打包+mask、Elephant 4D 动作切轴+多步路由、action header 可选 mask(~1 天)。实现顺序:先跑 K=4 滑动窗口冒烟基线,再实现本方案。
