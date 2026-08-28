#!/usr/bin/env bash
# =============================================================================
# 简单版自动测试：等待 epoch_0100.pt 生成后运行测试
# 
# 使用方法（在服务器上运行）:
#   cd /nas/liyangguang103/new_fscd/113_test
#   nohup bash wait_and_test.sh > wait_and_test.log 2>&1 &
#   
# 然后你可以放心去睡觉了！明天查看 wait_and_test.log 即可看到结果。
# =============================================================================

CKPT="/nas/liyangguang103/new_fscd/runs/mod_squad_10per_1_14_miloss_15_8_1/epoch_0100.pt"
THIS_DIR="/nas/liyangguang103/new_fscd/113_test"

echo "[$(date)] 开始监控权重文件: $CKPT"
echo "[$(date)] 每5分钟检查一次..."

# 等待文件出现
while [[ ! -f "$CKPT" ]]; do
    echo "[$(date)] 文件尚未生成，5分钟后再检查..."
    sleep 300
done

echo "[$(date)] ✓ 检测到权重文件！"
echo "[$(date)] 文件大小: $(ls -lh "$CKPT" | awk '{print $5}')"

# 等待30秒确保写入完成
echo "[$(date)] 等待30秒确保文件写入完成..."
sleep 30

echo "[$(date)] 开始运行测试..."
echo "=========================================="

# 运行测试
cd "$THIS_DIR"
bash test.sh "$CKPT"

echo "=========================================="
echo "[$(date)] 测试完成！"
