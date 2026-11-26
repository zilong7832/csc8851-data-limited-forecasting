# Team 17 · CSC 8851 服务器运行说明

本次运行会补齐 proposal 和中期报告里的最终实验，并为课堂展示生成可编辑数据表和高清图。两份 ETTh 数据已包含在包里。在已分配到 **一张 NVIDIA GPU** 的 Linux 节点运行，确保目录持久可写。

## 启动

解压 `Team17_GPU_Study.zip` 后进入 `Team17_GPU_Study` 目录。如果从 GitHub 克隆，进入仓库根目录。使用服务器提供的 CUDA 版 PyTorch 环境和 Python 3.10 或更新版本。

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python3 -m pip install -r requirements.txt
bash run_server.sh --gpu-check
bash run_server.sh
```

第一条命令的最后一项应为 `True`。如为 `False`，先进入已分配 GPU 的计算节点、激活服务器提供的 CUDA PyTorch 环境。学校集群上的 GPU 申请命令取决于集群配置。计算节点离线时，可以在同一共享环境的登录节点提前装好依赖。已有满足要求的库时可直接运行后两条命令。在服务器使用任务调度器时，把最后一条命令放在 GPU 作业脚本里。

脚本会用当前环境的 `python3`，也可以通过 `PYTHON=/path/to/python bash run_server.sh` 指定解释器。只做数据、窗口和模型形状检查可运行 `bash run_server.sh --check`，无须 GPU。常规整套运行会自动验证数据，并用极小的人造输入检验 CUDA 前向、反向和优化器。

## 中断与恢复

原目录、相同环境、相同 GPU 型号下再次运行 `bash run_server.sh`。已完成的模型会从检查点恢复，中断时正在进行的那一个模型会从头训练。训练和评估文件均在成功写完后才替换为正式文件。为获得可比耗时，请使用同一张空闲 GPU，避免和其他作业共享。建议在调度器内或已有的 tmux 会话中运行。需要持久化到其他磁盘时，从第一次开始使用同一个绝对路径：

```bash
bash run_server.sh --out /your/persistent/path/team17
```

## 跑完后回传

看到 `COMPLETE. Return this file:` 后，把它打印出的 **Team17_RESULTS.zip** 发回来即可。默认在 `runs/team17/Team17_RESULTS.zip`。自定义目录时，压缩包在对应目录。回传包会包含用于最终测试的全部 162 个模型检查点；以后想换一个测试日期画图，也可以直接加载模型。回传包里有所有终稿表格所需的核心数值、时间序列误差、示例预测、模型选择记录、按 epoch 的损失、GPU 信息及 PDF/PNG 图。中期报告的旧验证结果单独保留在仓库根目录的 `results/`。

出错时，保留已经生成的目录以及终端最后一段报错。脚本只在核对实验覆盖率、计算指标一致性、图片导出和压缩包完整性之后，才会发出完成提示。

## 各组实验在回答什么

| 问题 | 对照设计 | 主要输出 |
|---|---|---|
| 记录只有三个月时，注意力模型值得用吗？一年时呢？ | 两个站点单独训练。预测 24 和 96 小时，每组分别给 25%、50%、100% 历史，比较 proposal 的五个模型 | test_summary.csv、history_learning_curves |
| 精度提升需要付出多少计算成本？ | 同硬件记录最佳配置训练用时、参数量、停止轮次，并单独保留调参用时 | test_by_seed.csv、learning_rate_selection.csv、training_cost |
| 优势来自编码器深度还是处理温度漂移？ | ETTh1、24 小时、25% 和 100% 历史。一次只改一项。每项与主模型用相同学习率和种子配对 | paired_ablations.csv、transformer_ablations |
| 归一化究竟消除了哪种偏移？ | 固定模型，生成 512 条独立合成序列，比较原始条件、过去和未来同时偏移、仅未来突变三种场景 | simulation.csv、paired_shift_simulation |
| 面对实际曲线，模型具体错在哪里？ | 测试期起点和中点预先固定，示例包括真实温度和各模型预测；汇总每个预测步长和每个预测时点的误差 | forecast_examples.csv、forecast_context.json、by_forecast_lead.csv、origin_mae.csv.gz |
| 模型差距相对于时间波动有多大？ | 在相同测试时点配对，先平均各随机种子的绝对误差，再按连续时间块重采样 | paired_mae_differences.csv |

主要实验共 **210 次训练**：48 个“数据集 × 预测长度 × 历史比例 × 学习模型”组合各试两个学习率，复用选中的 seed 0，再补 seed 1、2，共 192 次；深度与归一化对照另加 18 次。共有 174 条测试记录，其中 12 条为确定性的“重复最后一天”基线。额外的“保持最后一个温度不变”只作为诊断参考，不需要训练。

展示组号为 **17**。最终报告按课程细则使用双栏 CVPR 格式，正文恰好四页，互评从第五页开始，其后为参考文献，可再附完整结果。成果文件待本次服务器结果返回后定稿。
