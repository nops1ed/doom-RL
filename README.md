
```bash
pip install -r requirements.txt
```

2. 安装ViZDoom（需要系统依赖）：
```bash
# macOS
brew install cmake boost boost-python3 sdl2
pip install vizdoom

# Ubuntu/Debian
sudo apt-get install cmake libboost-all-dev libsdl2-dev
pip install vizdoom
```

Train:
```bash
python train.py --episodes 1000 --save-dir checkpoints --log-dir logs
```

```bash
python train.py \
    --config configs/doom_basic.cfg \
    --episodes 2000 \
    --save-dir my_checkpoints \
    --log-dir my_logs \
    --render  # 可视化训练过程
```

Test:
```bash
python test.py --model checkpoints/best_model.pth --episodes 10 --render
```
```bash
python test.py \
    --model checkpoints/best_model.pth \
    --episodes 5 \
    --render \


```bash
python test.py \
    --model checkpoints/best_model.pth \
    --episodes 5 \
    --render \
    --save-video test_demo.avi
```

- **learning_rate**: 学习率 (默认: 1e-4)
- **batch_size**: 批次大小 (默认: 32)
- **gamma**: 折扣因子 (默认: 0.99)
- **epsilon_start**: 初始探索率 (默认: 1.0)
- **epsilon_min**: 最小探索率 (默认: 0.01)
- **buffer_size**: 经验回放缓冲区大小 (默认: 100000)
- **target_update**: 目标网络更新频率 (默认: 1000步)
