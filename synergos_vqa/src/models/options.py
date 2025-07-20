import argparse

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
    def parse(self):
        # 数据参数
        self.parser.add_argument('--data_dir', type=str, default='processed_data',
                               help='数据目录')
        self.parser.add_argument('--output_dir', type=str, default='outputs',
                               help='输出目录')
        
        # 模型参数
        self.parser.add_argument('--model_name', type=str, default='t5-base',
                               help='预训练模型名称')
        self.parser.add_argument('--max_length', type=int, default=512,
                               help='最大序列长度')
        self.parser.add_argument('--vis_feat_dim', type=int, default=256,
                               help='视觉特征维度')
        self.parser.add_argument('--hidden_dim', type=int, default=768,
                               help='隐藏层维度')
        
        # 训练参数
        self.parser.add_argument('--batch_size', type=int, default=16,
                               help='批次大小')
        self.parser.add_argument('--num_epochs', type=int, default=10,
                               help='训练轮数')
        self.parser.add_argument('--learning_rate', type=float, default=5e-5,
                               help='学习率')
        self.parser.add_argument('--weight_decay', type=float, default=0.01,
                               help='权重衰减')
        self.parser.add_argument('--num_workers', type=int, default=4,
                               help='数据加载线程数')
        self.parser.add_argument('--warmup_steps', type=int, default=1000,
                               help='预热步数')
        self.parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                               help='梯度累积步数')
        self.parser.add_argument('--max_grad_norm', type=float, default=1.0,
                               help='梯度裁剪阈值')
        
        # 生成参数
        self.parser.add_argument('--num_beams', type=int, default=4,
                               help='束搜索大小')
        self.parser.add_argument('--min_length', type=int, default=1,
                               help='最小生成长度')
        self.parser.add_argument('--max_length', type=int, default=50,
                               help='最大生成长度')
        self.parser.add_argument('--no_repeat_ngram_size', type=int, default=3,
                               help='n-gram重复惩罚大小')
        self.parser.add_argument('--early_stopping', type=bool, default=True,
                               help='是否使用早停')
        
        # 设备参数
        self.parser.add_argument('--device', type=str, default='cuda',
                               help='训练设备')
        self.parser.add_argument('--seed', type=int, default=42,
                               help='随机种子')
        
        # 日志参数
        self.parser.add_argument('--logging_steps', type=int, default=100,
                               help='日志记录步数')
        self.parser.add_argument('--save_steps', type=int, default=1000,
                               help='模型保存步数')
        self.parser.add_argument('--eval_steps', type=int, default=1000,
                               help='评估步数')
        
        return self.parser.parse_args() 