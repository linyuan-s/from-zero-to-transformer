import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tokenizers import Tokenizer
import os

# --- 1. 从 build_tokenizer.py 导入特殊标记 ---
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"

class TranslationDataset(Dataset):
    """PyTorch Dataset 用于翻译任务"""
    
    def __init__(self, split='train'):
        super().__init__()
        
        # 1. 加载 Hugging Face 数据集
        self.dataset = load_dataset("bentrevett/multi30k", split=split)
        
        # 2. 加载我们训练好的分词器
        tokenizer_dir = "tokenizers"
        self.src_tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer_en.json"))
        self.tgt_tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer_de.json"))
        
        # 3. 获取特殊标记的 ID
        self.src_pad_id = self.src_tokenizer.token_to_id(PAD_TOKEN)
        self.tgt_pad_id = self.tgt_tokenizer.token_to_id(PAD_TOKEN)
        
        self.src_bos_id = self.src_tokenizer.token_to_id(BOS_TOKEN)
        self.src_eos_id = self.src_tokenizer.token_to_id(EOS_TOKEN)
        self.tgt_bos_id = self.tgt_tokenizer.token_to_id(BOS_TOKEN)
        self.tgt_eos_id = self.tgt_tokenizer.token_to_id(EOS_TOKEN)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        根据索引获取一个数据对
        """
        # 1. 获取原始文本
        item = self.dataset[idx]
        src_text = item['en']
        tgt_text = item['de']
        
        # 2. 将文本分词并转换为 ID
        src_ids = self.src_tokenizer.encode(src_text).ids
        tgt_ids = self.tgt_tokenizer.encode(tgt_text).ids
        
        # 3. 添加 [BOS] 和 [EOS] 标记
        # 源序列 (Encoder 输入): [BOS] + ids + [EOS]
        src_ids = [self.src_bos_id] + src_ids + [self.src_eos_id]
        
        # 目标序列 (Decoder 输入/输出):
        # Decoder 输入: [BOS] + ids
        # Decoder 目标 (用于计算 loss): ids + [EOS]
        # 这里返回完整的 [BOS] + ids + [EOS]，在 collate_fn 中拆分
        tgt_ids = [self.tgt_bos_id] + tgt_ids + [self.tgt_eos_id]
        
        # 4. 转换为张量
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


class CollateFn:
    """
    可调用的类，用于 DataLoader 的 collate_fn
    它负责将一个批次 (batch) 的数据进行填充 (padding)
    """
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        """
        处理一个批次的数据
        batch: 一个列表，列表中的每个元素是 (src_tensor, tgt_tensor)
        """
        
        # 1. 分离 src 和 tgt
        src_batch = [item[0] for item in batch]
        tgt_batch = [item[1] for item in batch]
        
        # 2. 对 src 和 tgt 进行填充
        # pad_sequence 会自动将它们填充到该批次中最长序列的长度
        src_padded = pad_sequence(src_batch, 
                                  batch_first=True, 
                                  padding_value=self.pad_id)
        
        tgt_padded = pad_sequence(tgt_batch, 
                                  batch_first=True, 
                                  padding_value=self.pad_id)
        
        # (src_padded, tgt_padded) 形状均为 (batch_size, seq_len)
        return src_padded, tgt_padded


def get_data_loader(split, batch_size):
    """
    辅助函数：创建并返回一个 DataLoader
    """
    dataset = TranslationDataset(split=split)
    
    # collate_fn 实例，使用目标语言的 pad_id
    # (源和目标 pad_id 应该是一样的，但用 tgt 的更规范)
    collate_fn = CollateFn(pad_id=dataset.tgt_pad_id)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'), # 只在训练时打乱数据
        collate_fn=collate_fn
    )
    return dataset, data_loader

# --- 5. 测试代码  ---
if __name__ == "__main__":
    # 这个测试代码现在依赖于 utils.py 中的掩码函数
    from utils import create_padding_mask, create_combined_mask

    BATCH_SIZE = 4
    
    print("创建 DataLoader (test)...")
    test_dataset, test_loader = get_data_loader(split='test', batch_size=BATCH_SIZE)
    
    print(f"源 (en) 词表大小: {test_dataset.src_tokenizer.get_vocab_size()}")
    print(f"目标 (de) 词表大小: {test_dataset.tgt_tokenizer.get_vocab_size()}")
    print(f"源 <pad> ID: {test_dataset.src_pad_id}")
    print(f"目标 <pad> ID: {test_dataset.tgt_pad_id}")
    
    # 从 loader 中获取一个批次的数据
    src_batch, tgt_batch = next(iter(test_loader))
    
    print(f"\n--- 批次数据形状 ---")
    print(f"源 (src) 批次形状: {src_batch.shape}")
    print(f"目标 (tgt) 批次形状: {tgt_batch.shape}")
    
    # --- 准备模型的输入 ---
    # Transformer 的训练目标是 "给定前 i-1 个词，预测第 i 个词"
    
    # 1. Decoder 输入: <bos> + "A boy in a red hat..."
    # (即去掉最后一个 <eos> 标记)
    tgt_input = tgt_batch[:, :-1]
    
    # 2. 目标 (用于计算 loss): "A boy in a red hat..." + <eos>
    # (即去掉第一个 <bos> 标记)
    tgt_target = tgt_batch[:, 1:]
    
    print(f"\n--- 模型输入形状 ---")
    print(f"Decoder 输入 (tgt_input) 形状: {tgt_input.shape}")
    print(f"模型目标 (tgt_target) 形状: {tgt_target.shape}")
    
    # --- 创建掩码 ---
    print(f"\n--- 掩码测试 ---")
    src_mask = create_padding_mask(src_batch, test_dataset.src_pad_id)
    # 目标掩码是组合掩码，用于 Decoder 的自注意力
    tgt_mask = create_combined_mask(tgt_input, test_dataset.tgt_pad_id)
    # Encoder 的 padding mask 也会在 Decoder 的交叉注意力中使用
    
    print(f"源 padding mask (src_mask) 形状: {src_mask.shape}")
    print(f"目标组合 mask (tgt_mask) 形状: {tgt_mask.shape}")
    
    print("\n数据加载器和掩码准备就绪！")