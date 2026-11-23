# pairwise-aec-double-talk

## 领域

游戏、图形与多媒体

## 主题

回声消除双讲

## 说明

双讲检测、时钟漂移与近端保护。

初始状态下 `media/aec_double_talk.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
