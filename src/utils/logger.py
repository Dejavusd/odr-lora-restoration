"""
日志与指标记录工具。

包含：
    - get_logger: 创建同时输出到控制台和文件的 logger
    - MetricLogger: 记录训练/评估指标并保存为 JSON
"""

import os
import logging
import json
from datetime import datetime


# ============================================================
# 日志
# ============================================================

def get_logger(name, log_dir=None):
    """
    创建 logger。
    参数：
        name: logger 名称，同时作为日志文件名前缀
        log_dir: 日志文件保存目录，None 表示只输出到控制台
    返回：
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件 handler
    if log_dir is not None:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f'{name}.log')
            fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception as e:
            # 如果文件 handler 创建失败，至少保留控制台输出
            logger.warning(f"无法创建日志文件: {e}")

    # 避免向根 logger 传播，防止重复输出
    logger.propagate = False
    return logger


# ============================================================
# 指标记录
# ============================================================

class MetricLogger:
    """
    记录指标并保存为 JSON 文件。
    用法：
        ml = MetricLogger('./outputs/metrics.json')
        ml.log({'task': 'derain', 'epoch': 1, 'psnr': 30.5})
        ml.save()
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, item):
        """追加一条记录。"""
        if isinstance(item, dict):
            item = dict(item)
            item.setdefault('timestamp', datetime.now().isoformat())
        self.data.append(item)

    def extend(self, items):
        """批量追加记录。"""
        for item in items:
            self.log(item)

    def save(self):
        """保存到 JSON 文件。"""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存 metrics 失败: {e}")

    def clear(self):
        """清空内存中的记录。"""
        self.data = []

    def __len__(self):
        return len(self.data)


# ============================================================
# 导出
# ============================================================

__all__ = [
    "get_logger",
    "MetricLogger",
]