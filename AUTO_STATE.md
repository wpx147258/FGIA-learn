
[02:40] 远程 4×3080Ti（connect.nmb2.seetacloud.com:12179，/root/autodl-tmp/cub-bird）已接入：
       通过本地 -> 远程裸仓库（cub-bird.git）SSH 推送同步代码，批次脚本 scripts/remote_4gpu_batch1.sh。
       第一批并行：tl_replace / tl_js / content_pool / pa4_js（各卡约 8.6GB、85-96%）。
       本地 3090 同时跑 tl_attn_d1 起的对照矩阵。
[03:20] 远程第一批出分：tl_replace 88.68 / tl_js 88.68 / content_pool 88.85 / pa4_js 88.82
       （配对检验全部未检出差异，参照 r50_base_repro 89.06）。
[03:45] 本地 queuePP：tl_attn_d1 88.18（−0.88，显著更差）、se_fixed 87.87（−1.19，显著更差）、
       attnp_cls 88.99（−0.07，持平）。SE 线关闭。
[03:47] 关键更正：把 tl_replace 的权重拉到本地跑 probe_token_stats，八张图 JS=0.000、
       中心化余弦 1.000、IoU 1.000、熵 4.77（均匀的 90%）——**替换出口无正则时就是 GAP**，
       之前 1.66× 的富集比是 GAP 自带的聚焦。判据随之改成「富集比须明显高于 1.66×」。
[03:50] batch6 前 10 epoch 的结构性发现：replace + 双侧熵目标 ep3 只有 52%（基线 79），
       而同样正则放残差出口几乎无影响；熵初始化 = ln196 ≈ 5.28，目标 2.0 的惩罚 ≈10.8，
       replace 出口下正则梯度**直接改写分类输入**。→ tl2 改成四个 floor 格子（已替换脚本）。
[03:52] 新机制落地：TRAIN.tl_aux_w —— 每个 token 单独过同一个分类头算 CE（权重绑定，
       不增参数）。把部件锚在可分类区域上，堵住「分化到背景」。本地 queueQQ 与远程
       batch3（改版）都在测它；默认 0，不影响已有实验。
[04:02] 本地抢占成功：tl_cls_d2 刚起头就被停掉，queueQQ 接管（TL 正则消融 + aux CE）。
       tl_cls_d1 出分 88.85（−0.21，未检出）；结合 tl_attn_d1 的 −0.88，说明 TL 上
       **读出方式比注意力深度更关键**。
[04:05] 富集比探针修正：补上 Resize(512)+CenterCrop(448) 的几何映射后，GT 框面积 0.412→0.603，
       各实验富集比整体下调（GAP 1.66→1.37、tl_js 0.32→0.58、pa4 1.22→1.13）。结论方向不变。
[04:06] 新开关 tl_js_reduce=pair：tl_floor 的 train_tl_reg 到 ep15 掉到 1e-4，说明旧写法
       hinge(图对平均 JS) 可以被「少数图对拉开、其余是副本」满足；pair 版对每张图各自 hinge。
[04:30] batch6 出分：rep_js 85.81(−3.24)、rep_js_w01 85.24(−3.81)、js_w01 88.76(−0.29)、
       floor 88.49(−0.57)。关键：富集比最高的 rep_js_w01(1.63×) 准确率最差——「部件落在鸟上」
       在已扫参数下未产生增益；mean 读出与 GAP 同构，只重分配不加信息。→ 下一步做 concat 读出。
[04:35] token 统计补齐：tl_rep_js_w01 的 1.63× 是「一张好图复制 8 次」（JS=0、IoU=1、副本占比 1.0）；
       tl_floor 八张图真的不同（JS 0.405、IoU 0.274、副本 0）但富集比只有 1.11×、准确率 −0.57。
       → 分化、落点、增益三者在这条线上从未同时成立；concat 读出是最后一条没试过的结构环节。
[04:40] 判决性诊断（probe_token_probe，tl_floor 权重，线性探针 2000/2000）：
       gap 87.60 / mean 85.75 / concat 85.50；单 token 四个 ~85、四个 ~10（半数死掉）。
       → 部件之间**没有互补信息**（concat−mean=−0.25），且 mean 比 GAP 低 1.85。
       含义：换读出救不了这条线；要做的是「让每个 token 都可靠」（aux CE / 可靠性门控 /
       部位关键点监督）。batch4 的 concat 仍跑一轮做直接确认。
[04:41] r50_tl_rep_aux 出分 88.47（−0.59，p=0.032）。线性探针：8 个 token 全部活（86.7~86.8，
       原来 4 个接近随机），但完全相同（富集 1.37×=GAP 水平），concat−mean 仅 +0.20。
       → 辅助 CE 解决了「死 token」，但没解决「冗余」；关键是它与 JS 的组合格。
[05:00] 独立审查（subagent，346k tokens）结论：
       1) 现有合法单视图 R50/448 的公开上限是 PMG 89.6（单阶段约 88.9），我们 89.06 已接近；
          关键点监督在强基线上的合理预期 +0.2~0.5，破 90 概率 <10~20%；
       2) 唯一「增加信息」的合法通道是**训练期特权裁剪**（cam_crop 是本项目唯一稳定正项 +0.79，
          NTS-Net K=4 vs K=0 +2.2 全部来自多区域重编码），而注意力重加权与 GAP 同构；
       3) 关键点热图 + 自注意力大概率只是把监督挪个地方（14×14 上 15 个部位平均 0.94 个重叠），
          只值得作为证伪实验；
       4) 用 CUB 自带关键点不违反硬约束，但要把结果单独标成 part-supervised；
       5) 我的线性探针结论被判定为过度解读，已在 §14.1 收紧口径。
       → 行动：本地 queueRR 跑 4 个 gt_head 特权裁剪格子；同时修了「gt_head 在 cutmix batch
         被整体跳过」的 bug（否则七成监督被砍）。
[05:05] 多部位特权裁剪落地（代码 + 数据都验过）：
       dataset 按关键点裁 wing(9,13)/tail(14)（缺失退回 body 框以保证批内键一致），
       CamAux 每路共享骨干重编码 + 硬标签 CE（TRAIN.cam_part_views / cam_part_w），
       评估路径不经过。抽样核对过 head/body/wing/tail 的裁剪图，几何正确。
       远程 batch4 已换成四格：gt_head / +body / +wing / +wing+tail（原 concat 轮存档为
       scripts/remote_batch4_concat_later.sh，按审查意见暂缓：concat−mean 只有 +0.20）。
[05:00] batch4 定为「单视图 gt_head 的四种配置」而不是多部位：显存账算不过来。
       bs2/448 四路前向实测峰值 3.37GiB，按此推算 bs16 会到 12GB 上限附近；
       基线两路前向就已 8.6GB。多部位版要么梯度累积（micro-batch 8×2）、要么
       给辅助分支上 checkpoint（但 BN 统计会被重算两次），都先不动，等单视图结果。
       batch4 = hl / soft / c05 / nocons，与本地 queueRR 完全同名同参 → 两边合起来
       就是一配置两 seed，正好量 seed 方差。
[05:20] tl2 出分：rep_floor 88.19(−0.86)、j015 88.32(−0.71)、floor15 88.16(−0.86)、s4 88.42(−0.64)，
       四格全部显著更差。规律：JS 越强越亏（S=4 四张副本 88.42 反而最高），富集比全部≈GAP 的 1.37×。
       → replace 出口在已扫组合里最好 88.19；残差出口最好 88.76(−0.29, n.s.)。参数未穷尽。
       决策：远程杀死 batch3（TL+aux 组合，同一个必输家族），4 卡立刻切到 GT 特权裁剪。
[05:21] floor_aux 出分 88.06（−1.00，p=0.0005），富集比 1.35×。token 统计：熵 4.19、JS 0.329、
       IoU 0.268、副本 0、辅助 CE 0.85 —— 机制完全按设计工作（8 个部件互不相同且各自可分类），
       仍显著更差。TL 在已扫范围内未追平；本地 GPU 转去 gt_head 四格。
[06:10] gt_head_hl 出分 88.56（−0.50，p=0.10，未检出）。GT 头框（3.9% 面积）没赢 CAM 框（10.6%）：
       裁剪分支 CE 1.7~2.1 vs CAM 1.2，头框太紧、缺颈胸上下文。
       本地已抢占切到 part_pad=0.8（面积 9.3%）两格；远程 batch4 其余三格在跑。
[06:20] batch4 四格：hl 88.61（与本地 88.56 差 0.05，两 seed 一致）、soft 88.75、c05 88.45、
       nocons 88.30（p=0.012 显著更差）。→ 一致性权重 0.3 最优、0 最差，说明裁剪分支的
       价值主要来自与主分支的一致性；GT 头框整组仍低于 CAM 框基线 0.3~0.8。
[06:59] gth_pad08 出分 88.87（−0.19，p=0.55，未检出）——外扩把 GT 头框从 −0.45 拉到 −0.19，
       是该家族最好的一格，但仍只是持平：GT 框不比 CAM 框提供更多有效分辨率。
       本地已接 pad08 + 裁剪 CE 0.4；远程 batch5 四格在 ep32 左右。
[07:25] batch5 出分：cam_head_w03 88.76、cam_head_w05 88.78、gth_wing 88.71、cam_wing_w03 88.47，
       四格全部未检出差异、全部低于 89.06。CAM 框之上叠加 GT 部位视图没有增益。
       → 远程已启动 batch6（组合与收尾：部位一致性 / pad08+翅 / pad08+AttnPool / 一致性 0.4）。
[07:48] gth_pad08_w04（裁剪 CE 0.4）88.61（−0.45），比 CE 0.2 的 88.87 差 0.26。→ GT 头视图
       权重不该超过基线 CAM 框的 0.2。特权裁剪家族的最好成绩仍是 88.87（−0.19，n.s.）。
[07:53] 发现并修复一个启动即崩的 bug：`MODEL.attn_dropout: null` 时
       `float(mcfg.get("attn_dropout", mcfg.get("dropout", 0.0)))` 返回 None 而崩
       （.get 的默认值只在键不存在时生效，键存在但为 null 时不生效）。
       batch6 的 r50_gth_pad08_attn 因此没跑起来，已在 GPU2 单独重启（其余三格正常）。
       本地 queueTT 的 cam_head_pc 已在 07:49 起跑（之前的 watcher 有 80 分钟上限，
       超时退出导致 GPU 空了几分钟——之后 watcher 一律给足时长）。
[08:35] batch6 前两格：cam_head_w05_pc 88.51（−0.55，加部位一致性反而更差）、
       cam_head_w05_c04 88.82（−0.24，一致性 0.4≈0.3 的平台）。
       → 「CAM 一致性有效是因为 CAM 框来自当前模型（自洽、会移动），GT 固定标注没有这个性质」。
       剩余两格（gth_pad08_wing、gth_pad08_attn）9 点后出。
[08:33] 第一个超过基线的配置出现：r50_gth_pad08_wing（GT 头 pad0.8 448 + GT 翅 224，CE 0.3）
       test 89.16（+0.10，p=0.76 未检出）——多部位特权裁剪，方向与 NTS-Net 一致。
       立即追加：远程 GPU3 头+翅+尾、GPU1 头+翅(CE 0.5)；本地排队两个 seed 重复
       （SEED=7/13，同配置），回答「+0.10 是真是假」。GPU0 保留 100 epoch 基线，
       用来判断 50 epoch 是否欠训练。
[09:21] 按用户要求**暂停强监督**（GT 关键点特权裁剪）：远程 gth_wt / gth_w_w05 已停，
       本地 seed 重复（gth_w_s2）已停，不再在关键点标注上投入。
       转回 pa4（弱监督、纯图像标签）的调优：
       - 本地 GPU：r50_pa4_l3_k4（layer3 出掩码，K=4）
       - 远程 GPU1：r50_pa4_k8_s1（K=8 的 SEED=1，判定 89.21 真伪）
       - 远程 GPU3：r50_pa4_l3_k8（layer3，K=8）
       - 远程 GPU0/2：两个 TL「压紧+锚」格继续（ent3 + aux，替换/残差）。
       注意：pa_div_w / pa_aux_w / pa_lr 在 TRAIN 下，pa_from/pa_parts/pa_tau 在 MODEL 下，
       写错会直接 KeyError（今天踩了一次）。
[09:45] 核对「原计划是否都恢复了」：把 scripts/queue_r50_90*.sh 里所有 run 名与 outputs 对照，
       发现原计划有 11 项从未执行（不只是 gc/clip 四格）：
         · 第一轮（已挂等待）：r50_gc、r50_gc_ctrl、r50_clip_gap、r50_clip_ap（+ s8_gc 视 gc 结果）
         · 控制臂/消融：r50_pa、r50_pa_ctrl、r50_pa2_ctrl、r50_pa3_ctrl
         · 多区域：r50_camfuse3（三路）、r50_hypfuse2、r50_rerank2、r50_rerank4
         · 变体：r50_pa4_lr1、r50_pa4_logit_s1、r50_partaux
       已写成三轮脚本 /root/restore_plan.sh，并由 /root/chain_restore.sh 挂到第一轮之后自动执行。
       注意 r50_rerank2_ctrl 在原队列里与 rerank2 配置完全相同（疑似复制粘贴错误），未纳入。
       强监督项（gth_w_s3）按用户要求继续暂停，不补。
[09:50] 用户提醒：OPTIMIZATION_BACKLOG 的 P0/P1 对齐臂也没跑。核对确认：
       · 「dropout 对齐」（attn_dropout=0 / tl_dropout=0）—— AttnPool/TL 用 0.1，
         PartAttention/GlobalContext 用 0，跨族比较一直带着这个混淆，从未做对齐臂；
       · attn_tokens=49（只有 10/26 的四视图版 89.13，单视图没跑）；
       · attn_depth=0 控制臂（单视图没跑）；
       · attn_pos=true、attn_dim=512/heads=8（P1，未跑）。
       已加第 E 轮（/root/restore_phaseE.sh）：attnp_t49_sv、attnp_d0_sv、attnp_nodrop、tl_nodrop，
       挂在 restore_plan 之后自动执行。
