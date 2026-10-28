
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
