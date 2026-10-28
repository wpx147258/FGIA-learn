自动推进 playbook

给定时任务用的操作手册。每次触发按这里走，不要临场发挥。

============================================================
零、硬约束（任何情况下都不许违反）
============================================================
1. 一次只跑一个 run。本机只有一张 3090。
2. 基准配方固定，除了「本轮要测的那一个变量」之外，一个字节都不改：

     DATASET.image_size=448 TRAIN.batch_size=16 TRAIN.epochs=50
     TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
     TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
     TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0
     TRAIN.backbone_lr_layers.stem=0.01 … layer1=0.01 layer2=0.03 layer3=0.05 layer4=0.05
     MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1
     TRAIN.swa_start_frac=[0.5,0.6,0.7]

3. 不删除、不覆盖任何已有 outputs 目录。新的 EXP_NAME 必须唯一。
4. 不修改模型/数据/损失代码。只允许改 --set 参数。需要改代码的实验一律记为 BLOCKED，跳过。
5. 启动用 setsid nohup，日志写 logs/<name>.log，确认 40 秒内出现 epoch= 行才算启动成功。
6. 单次触发最多启动 1 个 run。**整晚不设 run 数上限**（机器整夜不休、不关机）。
7. 参数扫小步。任何参数的分支上，两个相邻取值之间不要超过 1.5 倍
   （例如 k：12 → 15 → 18，不要 15 → 30；γ：0.05 → 0.08 → 0.12，不要 0.1 → 0.3）。
   粗跳只有在"已知没有信号、准备放弃这条线"时才允许。

============================================================
一、参照系与显著性（2026-10-24 按方法学审查修订）
============================================================
⚠️ 2026-10-24 用户约束（两条）：
   (1) **不允许集成**；
   (2) **train_on_val 不算数**（把 val 并回训练集属于取巧，不是因为机制变好）。
       → 所有实验一律在原始 5395 张上做，DATASET.train_on_val 必须为 false。
       alltrain 系列的结果只作为历史记录留在 AUTO_STATE，不进结论、不进报告。
多模型概率平均、多 checkpoint 平均都不算数。
   主指标改为「单个权重文件（swa_f0.5）+ 四个测试视图的 TTA」，也就是 test_tta.py
   输出里的第一行「该权重的 4 视图融合」。含 3 个 SWA 权重的 12 路融合只作参考，不作结论。

基准（原配方，单权重 + 四视图 TTA）：89.40
test n=5794，p≈0.8935：

    单次 top1 的标准误 0.41 个点，95% 区间 ±0.79
    两次独立训练差值的 SE ≈ 0.57 个点
    因此：**任何小于 0.5 个点的差异都不予采信**，
    要声称"提升"需要 ≥ 0.8 个点，并且配对检验 p<0.05。

⚠️ test 已经被用来比较过 20+ 个配置。每次拿最大值，选择偏差约 0.5~0.9 个点。
所以"历史最好 89.42"只能写成"参与选择后的上界"，不能当无偏成绩。

新的判定方式：
  1. 每测一个配置，都用 scripts/test_tta.py --dump 存逐样本概率
  2. 和基准比，用 scripts/compare_runs.py 做 McNemar + bootstrap 区间
  3. 只有 p<0.05 且 Δ 的 95% 区间不含 0，才记为"有差异"
  4. 否则一律记"未检出差异"，不要写成"没影响"或"等价"

val（599 张）的标准误 1.28 个点，只能分辨 2 个点以上的差异，只用作粗筛。

============================================================
二、状态机
============================================================
S1 检查 ps -ef | grep "train.py --config"（排除 grep 自身）
    有进程 → 记一行进度（最后一个 epoch= 的 val_top1）到 AUTO_STATE.md，结束本轮
    无进程 → 进入 S2

S2 找出「已训练完成但还没评测」的 run：
    该 run 目录里有 swa_f0.5.pth，且 AUTO_STATE.md 里没有它的 test 记录
    有 → 进入 S3；没有 → 进入 S4

S3 评测（约 6 分钟）：
    /home/wpx/anaconda3/bin/python scripts/test_tta.py \
      --checkpoint <run>/swa_f0.5.pth <run>/swa_f0.6.pth <run>/swa_f0.7.pth
    取最后一行「总融合」的 top1，追加到 AUTO_STATE.md，然后进入 S4

S4 选下一个实验：看 AUTO_STATE.md 里已完成实验的序列，按下面第三节的决策树选，
    启动，在 AUTO_STATE.md 记「已启动」。

============================================================
三、实验队列与决策树
============================================================
分支 Z（优先级最高，先做完）：噪声基线
  Z1 SEED=1    EXP_NAME=swsl_448_seed1     （配方与基准完全一致，只换种子）
  Z2 SEED=2    EXP_NAME=swsl_448_seed2
  目的：现在所有判断都建立在一个没量过的数字上——同一配方换个随机种子会差多少。
  已有的 swsl_448_best_drop10（SEED=42）算第三个样本。
  三个跑完后：算它们的 12 路融合 test top1 的样本标准差 s，
  然后把第一节的判定阈值改成「±2s」，并把 s 写进 AUTO_STATE.md 顶部。
  在 s 出来之前，任何 ±0.3 以内的判断都只记为"未定"，不要写结论。

分支 A：硬负样本 CE（已扫 k=15、30，均无信号，暂缓）
  A1 k=15   EXP_NAME=swsl_448_gce15    89.35
  A2 k=30   EXP_NAME=swsl_448_gce30    89.16
  若要回来：小步走 k=12 和 k=18，不要再跳 2 倍。

分支 B：熵正则
  B1 γ=0.1                         EXP_NAME=swsl_448_maxent01
  B2 γ=0.05                        EXP_NAME=swsl_448_maxent005
  B3 γ=0.2                         EXP_NAME=swsl_448_maxent02
  （先 0.1 看方向；有正信号就 0.05 和 0.2 各一步，不要跳到 0.3）

分支 C：把 val 并回训练集（数据量 +11%）
  C1 DATASET.train_on_val=true      EXP_NAME=swsl_448_alltrain
  说明：599 张 val 本来就是从官方 train 里切出来的，合回去就是完整的 5994 张。
  代价是 val 指标从此没有意义（模型背过那些图），只看 test + SWA。
  不改模型、不改损失，纯数据变量，预期 +0.2~0.4；若有效可与其它结论直接叠。

分支 D：混合概率细化（在 C 的结论出来之后）
  D1 mixup_p=0.8   D2 mixup_p=0.6   —— 小步，各 ±0.1

分支 E：GeM 池化（可学习广义平均，f=(mean(x^p))^(1/p)）
  E1 DATASET.train_on_val=true MODEL.pool=gem MODEL.gem_p=3   EXP_NAME=swsl_448_alltrain_gem
     在 alltrain 基座上换池化。只多一个标量参数（p），其余配方不动。
     判据：单权重 + 四视图 TTA 能否超过 alltrain 的 89.96。

分支 F：head_norm（分类头前加 BatchNorm1d，config 里一直有这个开关但没在最终配方上试过）
  F1 DATASET.train_on_val=true MODEL.head_norm=true          EXP_NAME=swsl_448_alltrain_hn

优先级：Z（噪声基线，已完成）→ C（数据量，已完成 +0.52）→ E（GeM）→ F（head_norm）
        → B（熵正则，γ=0.05 在跑）→ A（GCE 细扫）→ D

决策：
  A1 结果与基准做配对检验，未检出差异 → 分支 A 暂缓（不是"判死"，是"未检出"）
  A2 同上
  若要回到分支 A：小步走 k=12 和 k=18，不要再跳 2 倍。
  B1 ≥ 89.65 → 跑 B2 和 B3 各一次（小步，两个方向都要）
  B1 在噪声带内 → 跑 B2（γ 往小的方向走一步），再按 B2 结果决定 B3
  B1 ≤ 88.95 → 走完 B2 再停（不要因为一次掉点就判死，小步扫完再说）

任何一个分支连续两个 run 都 ≤ 88.95，才放弃这条线。
不设 run 总数上限；但如果某个分支已经扫完所有小步都没有 ≥ 89.65 的结果，
就在 AUTO_STATE.md 里写"该分支无信号"，转下一条线。

============================================================
四、启动模板
============================================================
复制 logs/run_gce.sh，把 --set 里那个变量换成本轮的值、把 EXP_NAME 换成新的：

  sed -e 's|TRAIN.hard_neg_k=15|TRAIN.hard_neg_k=30|' \
      -e 's|EXP_NAME=swsl_448_gce15|EXP_NAME=swsl_448_gce30|' \
      logs/run_gce.sh > logs/run_gce30.sh
  chmod +x logs/run_gce30.sh
  cd /home/wpx/main/work/my-notes/projects/cub-bird
  setsid nohup bash logs/run_gce30.sh > logs/gce30_train.log 2>&1 < /dev/null &

熵正则的 run 用同样的模板，把 TRAIN.hard_neg_k=15 换成 TRAIN.entropy_w=0.1。

============================================================
五、异常处理
============================================================
- 日志出现 Traceback 或 CUDA out of memory → 记为 FAILED，不重试第二次，
  直接跳到下一个实验；如果连续两个 FAILED，停止并写总结。
- GPU 显存被别的进程占着（非本项目 train.py）→ 不启动，记一行，结束本轮。
- 训练跑完但缺 swa_f0.5.pth → 记 BROKEN，跳到下一个实验。
- 任何不确定的情况：不启动，写一行原因，结束本轮。宁可空转一轮，不要乱试。

============================================================
五之前、审查门（2026-10-26 用户指示）

  凡是"重要决定"，执行前必须先派独立的 subagent 做对抗审查，再动手。重要决定指：
    - 新增一条实验主线（比如 O 组自注意力这种要写代码的结构改动）
    - 改训练框架的关键路径（优化器参数分组、损失接线、checkpoint 语义）
    - 下"这个方向没用"的结论并据此关掉一条线
    - 写进报告的定量结论

  审查做法：
    - 至少两个不同视角（例如"机制与证据" / "实验设计与混淆"），不要两个都问同一个问题
    - brief 里明确写"只报问题、不总结优点"，并给输出结构（裁决 + 理由 + 必须的修改）
    - 审查结论落进 AUTO_STATE，不是只在对话里回一句
  依据：这几天靠审查抓出过真问题——FPN 会静默跳过 CAM 分支、stride 不在 state_dict、
  以及"新模块参数没进优化器"的静默失效，自查都没发现。

六之前、记录要求（2026-10-26 用户指示）

  目的不是涨点，是搞清楚"什么样的手段能涨点"。所以每条实验的记录必须包含三段：
    1) 数字：单权重四视图、相对 89.40 的差、配对检验的 p 与区间
    2) 机制解释：这个结果说明该类方法在我们这个设定下为什么有效/无效。
       要具体到机制，不能写"没有提升"四个字了事。
    3) 排除或确认了什么：这条结果让哪个假设出局、让哪个假设还活着。
  只记数字不写分析，视为该轮未完成。

六、收尾
============================================================
2026-10-25 新增分支 I（弱-强视图，已过审查与诊断门，代码已写好）：

  预注册（不许事后改）：
    λ_kl = 0.3，T = 2.0（与现有 cam_consistency 同值，便于对照）
    强视图：张量级几何扰动（旋转 ±15°、剪切 ±10°、平移 ±10%），从**混图之后**生成
    弱视图必须与 89.40 逐字节一致：启动后核对 config 里 rand_augment==''、random_erasing==0
    主对比：A3 vs A1（一致性在增强之外是否还有贡献）；次对比：A3 vs 89.40
    判据：|Δ|<0.3 → 关闭，不再扫权重；≥+0.5 且配对 p<0.05 → 再用 SEED=1 复现一次

  I1 A1 归因臂（强视图只当增强，无 KL）
     TRAIN.strong_view=true TRAIN.strong_ce=1.0 TRAIN.strong_kl=0.0   EXP_NAME=swsl_448_sv_a1
  I2 A3 完整（强视图 CE + 非对称 KL）
     TRAIN.strong_view=true TRAIN.strong_ce=1.0 TRAIN.strong_kl=0.3 TRAIN.strong_T=2.0
     EXP_NAME=swsl_448_sv_a3

  诊断门已完成（scripts/gate_invariance.py）：clean 89.06 → geom 85.71（净损失 3.35），
  门槛 ≥2.0 通过 —— 模型确实缺乏对几何扰动的稳健性，机制前提成立。

  已 DROP 的候选（有硬数据，不要重开）：
    C1 FPN 1×1 / C2 layer4 步长：layer3 GAP 线性探针只有 50.74，layer3+layer4 拼接 85.17，
      都低于 layer4 单独的 85.99 → 多尺度没有可读信息。
    B 图内两部位比较：两票 REVISE，且多区域拼接探针 86.66 < 全局 87.33。

2026-10-25 分支 H（余弦头）：已跑，88.04，显著更差，关闭。
  H1 MODEL.head=cosine   EXP_NAME=swsl_448_cos
     余弦分类头：特征和权重都 L2 归一化，logits = s·cos(θ)，砍掉模长自由度。
     依据：沿分类器方向 d′ 训练集 3.1~4.3、测试集 1.1~1.8，
     而类内散度/类中心几乎不变 → 问题在分类器可用的自由度，不在特征。
  H2 若 H1 有正信号：MODEL.dropout 0.1→0.15（余弦头下正则强度可能要重调）

2026-10-25 第二批候选审查结果（队列再次为空）：
  J1 SAM —— 两票 REVISE，但按审查要求做的锐度门失败了：ρ=0.01 下三个随机方向的
     Δtop1 = 21.45 / 7.35 / 17.07，方向间 sd≈7 点，无法用来给 checkpoint 排序。
     结论：不是机制被证伪，是**无法测量**，故不排队。
  J2 头部 wd 单独加大 —— 两票 DROP（读出头侧已饱和；头行范数分布均匀，
     均匀收缩对 argmax 近乎无效；同族已有四个负结果）。

2026-10-25 用户放开主干约束：可以换强 ViT / CLIP 主干（原约束是固定 ResNet-50）。
  依据：测量三已证明瓶颈在特征侧，换主干是唯一能动特征的杠杆。

  约束变化后的落地事实（已实测）：
    - CLIP ViT-B/16 @448/bs16 + cam_crop：15.3 GB，1.4 s/step → 50 轮约 6.6 h，25 轮约 3.2 h
    - ConvNeXt-B @448/bs16 + cam_crop：OOM（>22.8 GB），暂不排
    - Swin-B：位置编码固定 384，448 不可用，暂不排
    - ViT 的 token 序列已在本会话接通（classifier._to_feature_map），CAM 裁剪分支可用

分支 K（主干替换，队列）：
  K1 CLIP ViT-B/16，448，bs16，25 轮，head lr 5e-4（ViT 比 R50 敏感，降一半），
     warmup 5，drop_path 0.1，其余配方不变            EXP_NAME=vitb_clip_e25
     判据：单权重四视图 vs 89.40；≥+0.5 且 p<0.05 才继续 K2
  K2 若 K1 有正信号：同配置 50 轮（约 6.6 h）          EXP_NAME=vitb_clip_e50
  K3 若 K1 明显更好但没到位：head lr 5e-4→3e-4 小步    EXP_NAME=vitb_clip_lr3

  注意：换主干后 89.40 不再是同配方基线，但仍是"当前最好成绩"的参照；
        主判据是"能不能超过 89.40"，不是"相对 R50 同配方涨多少"。

分支 L：PMG（渐进多粒度拼图，Du et al. ECCV 2020）—— 按用户要求排在 ViT 之后

  已按官方仓库（PRIS-CV/PMG-Progressive-Multi-Granularity-Training）实现：
    model/pmg.py：jigsaw(n) 生成器 + 三路 conv_block + 四个分类器
    classifier：训练时每 batch 跑 n=8/4/2/原图 四次前向；推理只用原图，走 concat 分类器
    损失：main_w=2（concat）+ 三路 stage 各 1（官方设置）
    ⚠️ PMG 会强制关闭 CAM 辅助分支（官方也不用；开着会变成六次前向）
       → 这一项与 89.40 不是单变量对比：少了 cam_crop 0.2 + cam_consistency 0.3。
         判读时必须记住这一点，不能简单说"PMG 不如 R50 基线"。

  L1 MODEL.pmg=true TRAIN.main_w=2.0 TRAIN.pmg_stage_w=1.0
     TRAIN.cam_crop=0 TRAIN.cam_consistency=0       EXP_NAME=pmg_r50_e50
     参数 +19.6M，每步四次前向（约 2.5× 时间），预计 1.5~2 小时

  判据：单权重四视图 vs 89.40。同时记录"PMG 去掉 CAM 两件套"这件事，
        如果 PMG 在 88~89 之间，无法区分是 PMG 无效还是丢了那 +0.8。

分支 M：ViT 配方微调（2026-10-25 新开，当前最好成绩 91.66 在 ViT 上）

  背景：ViT 50 轮 = 91.66，25 轮 = 91.30，R50 = 89.40。瓶颈已确认在特征，主力应转到 ViT。
  这三条都是单变量、各约 1 小时，按优先级排队：

  M1 drop_path 0.1 → 0.2        EXP_NAME=vitb_dp20
     ViT 对随机深度比 CNN 敏感，0.1 是从 R50 配方沿用来的，没在 ViT 上验过。
  M2 head lr 5e-4 → 3e-4        EXP_NAME=vitb_lr3
     ViT-B 微调常用区间更低；配合分层 lr，layer4 会变成 1.5e-5。
  M3 cam_consistency 0.3 → 0.15  EXP_NAME=vitb_cons015
     这套机制在 R50 上值 +0.79，在 ViT 上从没单独验过权重。
  M4 head lr 5e-4 → 8e-4        EXP_NAME=vitb_lr8
     3e-4 已经测过（91.39，未检出）；往另一个方向再走一步，确认 5e-4 是否在峰上。
  M5 epochs 50 → 75             EXP_NAME=vitb_e75
     25→50 轮有 +0.36 的微弱增益，75 轮看是否还有余量。约 2.5 小时。

  判据一律：单权重四视图，与 91.66 配对 McNemar；|Δ|<0.3 关闭该项，不扫第二个值。

  PMG(R50) 已经跑过：82.02，显著更差，但那是 50 轮 vs 官方 300 轮的训练量差距，
  不是机制被证伪。除非用户明确要求跑 12 小时的长训练，不再排 PMG。

2026-10-26 07:10 主干这条线走完，结论：容量收益在 ViT-B 处饱和
  R50(24M) 89.40 → ViT-B(86M) 91.66（+2.26，显著）→ ViT-L(300M, bs8) 91.84（+0.18，未检出）
  ViT 配方四个旋钮（轮数 25/50/75、lr 3e-4/5e-4/8e-4、drop_path 0.1/0.2、cons 0.15/0.3）
  全部以 91.66 为峰。权重再换（SigLIP 90.54、OpenAI CLIP 91.59）也不超过它。

2026-10-26 用户指示：R50 上机制层面没有完全失效，继续做结构性实验，目的是练习涨点方式，
不追求结果。下面这一支的判据因此放宽为"记录结果与失效原因"，不以超过 89.40 为目标。

分支 N（R50 结构性实验）。全部代码已就绪，launcher 已写好，按顺序执行：

  N1 logs/run_r50_cbp.sh      EXP_NAME=r50_cbp_swsl      紧凑双线性池化（SWSL 上首测）
  N2 logs/run_r50_gapstd.sh   EXP_NAME=r50_gapstd_swsl   全局平均+标准差池化（同样只在 a1 上测过）
  N3 logs/run_r50_ds.sh       EXP_NAME=r50_ds_swsl       depth supervision，layer3 辅助分类器 0.3
  N4 logs/run_r50_se.sh       EXP_NAME=r50_se_swsl       SE 通道注意力，插 16 个 bottleneck，+2.4M 参数
  N5 logs/run_r50_distill.sh  EXP_NAME=r50_distill_vitb  从 ViT-B(91.66) 蒸馏到 R50，w=0.5 T=2

  执行方式：bash logs/run_r50_<名字>.sh 用 setsid nohup 起，日志同名。
  判据：单权重四视图，与 89.40 配对比较。**哪怕更差也记录结果与失效原因**，
        这一支的目的是练习不同的涨点方式，不以超过 89.40 为目标。

  注意 N5：教师是 ViT-B 的 swa_f0.5，训练时每步多一次 ViT 前向，约 1.5 倍时间。
        推理仍是纯 R50 单模型，不构成集成。

分支 O：CNN 特征图 -> token 序列 -> 轻量自注意力（2026-10-26 经两轮审查后重排）

  代码：model/attn_pool.py。输出与 GAP 同维相加，输出层零初始化。
  审查后的关键修正：
    - 修了 y=None 早退路径的 bug（CAM 裁剪分支原先不带 attn，导致一致性损失变质）
    - 加了兼容性断言（attn 不能与 pmg/fpn/fusion/吃特征图的头同开）与 token 数断言
    - 加了 TRAIN.attn_lr（attn 组独立学习率）与每轮 ‖out.weight‖ / 输出范数日志
    - 加了 depth=0 控制臂
  撤销 O3（layer3）：复测显示 layer3 的线性读出 train 96.2 / test 50.7，
  层越浅塌得越狠 —— 把注意力接在那里等于补偿一批不可用的特征，押注理由不成立。

  臂（按对照意图排列，一次只跑一条）：
    O0 logs/run_o0.sh   layer4 / 196 / d256 / depth=0 / 无PE     控制臂：没有注意力，
        只保留 1x1 投影 + token 平均 + 线性层。用来分离"多出来的参数与池化"和"注意力"。
    O1 logs/run_o1.sh   layer4 / 196 / d256 / 2 层 / 无PE        主臂
    O1b logs/run_o1b.sh 同 O1，但 TRAIN.attn_lr=3e-4             lr 校准臂
    O4 logs/run_o4.sh   layer4 / 49  / d256 / 2 层 / 无PE        token 数对照（用户指定）
    O2 logs/run_o2.sh   layer4 / 196 / d512 / 4 层 / 带PE        容量与位置编码（后期）

  预注册判据：O1 为主比较，与 89.40 配对 McNemar 并做 Holm 校正（共 4 次比较）。
  O1 vs O0 是"注意力是否带来增量"的直接对照，比各自比基线更有功效。
  若全部落在 ±0.3，只能写"未检出（检测下限约 ±0.4~0.6）"，不能写"无效"。

2026-10-27 10:50 用户指示：**不再做蒸馏**。分支 N5 及其全部变体（w/T/门控/换教师）到此关闭，
结果已记入 AUTO_STATE，不再排新的蒸馏实验。队列为空时只记一行，不要自行重启这条线。

跑满所有分支后：
  在 AUTO_STATE.md 末尾写一段「今晚小结」，列出每个 run 的配置、test、相对 89.35 的增量，
  并给出下一步建议。之后不再启动新实验。
