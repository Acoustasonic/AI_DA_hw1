# 围棋 AI 项目报告模板

## 1. 项目目标

简要说明本次作业的目标：
- 实现随机 AI
- 实现基于 MCTS 的围棋 AI
- 可选实现 Minimax + Alpha-Beta
- 设计一个人机对弈图形界面

## 2. 系统结构

说明项目主要模块及职责：
- `dlgo/`：围棋规则、棋盘、计分
- `agents/random_agent.py`：随机智能体
- `agents/mcts_agent.py`：MCTS 智能体
- `agents/minimax_agent.py`：Minimax 智能体
- `play.py`：命令行对局脚本
- `gui.py`：图形界面

## 3. 算法设计

### 3.1 随机 AI

说明随机 AI 的实现方法：
- 如何枚举合法着法
- 如何避免非法落子
- 是否保留 `pass` / `resign`

### 3.2 MCTS AI

介绍 MCTS 的四个阶段：
1. Selection
2. Expansion
3. Simulation
4. Backup

这里建议重点写：
- UCT 公式
- 节点统计量设计
- 最终选点策略

### 3.3 MCTS 优化

至少写两项优化，并说明原因：
- 启发式 rollout
- 限制模拟深度
- 对 `pass` 的终局处理
- 局部走子偏好或先验概率

### 3.4 Minimax（选做）

如果实现了该部分，可说明：
- 搜索深度
- Alpha-Beta 剪枝
- 局面评估函数
- 置换表缓存

## 4. 图形界面设计

说明 GUI 的主要功能：
- 显示棋盘和棋子
- 鼠标点击落子
- 支持新游戏
- 支持停一手
- 支持选择 AI 类型和执子方

可插入界面截图。

## 5. 实验设置

建议列出测试环境与参数：
- Python 版本
- 棋盘大小
- MCTS 模拟轮数
- Minimax 搜索深度
- 对局次数

## 6. 实验结果

### 6.1 MCTS vs Random

可用表格记录：

| 对局设置 | 黑方胜 | 白方胜 | 平局 | 平均步数 | 平均用时 |
| --- | --- | --- | --- | --- | --- |
| MCTS vs Random |  |  |  |  |  |

### 6.2 Minimax vs Random

| 对局设置 | 黑方胜 | 白方胜 | 平局 | 平均步数 | 平均用时 |
| --- | --- | --- | --- | --- | --- |
| Minimax vs Random |  |  |  |  |  |

### 6.3 MCTS vs Minimax

| 对局设置 | 黑方胜 | 白方胜 | 平局 | 平均步数 | 平均用时 |
| --- | --- | --- | --- | --- | --- |
| MCTS vs Minimax |  |  |  |  |  |

## 7. 结果分析

建议讨论：
- MCTS 为什么能优于随机策略
- rollout 优化是否有帮助
- Minimax 与 MCTS 的优缺点比较
- 5x5 小棋盘与真实围棋的差别

## 8. 与 AlphaGo / AlphaZero 的比较

可以从以下角度分析：
- 本作业 MCTS 与 AlphaGo/AlphaZero 中 MCTS 的差异
- 是否使用策略网络和价值网络
- 是否使用自对弈训练
- 搜索规模与效果差距

## 9. 总结

总结本次项目完成情况、收获与后续可改进方向。

## 10. 附录

可附：
- 关键代码片段
- 运行命令
- 对局截图
- 异常情况说明
