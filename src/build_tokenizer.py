import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# --- 1. 定义特殊标记 ---
# 定义词表中必须包含的特殊标记
PAD_TOKEN = "[PAD]" # 填充
UNK_TOKEN = "[UNK]" # 未知词
BOS_TOKEN = "[BOS]" # 句子开始
EOS_TOKEN = "[EOS]" # 句子结尾

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# 目标词汇表大小
VOCAB_SIZE = 30000 

# --- 2. 准备数据迭代器 ---

def get_batch_iterator(dataset, lang, batch_size=1000):
    """
    一个生成器函数，用于从数据集中批量产出文本。
    这是分词器训练所必需的。
    """
    # 我们只在训练集上训练分词器
    for i in range(0, len(dataset['train']), batch_size):
        batch = dataset['train'][i : i + batch_size]
        yield batch[lang]

# --- 3. 训练并保存分词器 ---

def train_and_save_tokenizer(lang, dataset_iterator):
    """
    为指定语言训练并保存一个 WordPiece 分词器
    """
    print(f"开始训练 {lang} 分词器...")

    # 1. 初始化 Tokenizer
    # 使用 WordPiece 模型，就像 BERT 一样
    tokenizer = Tokenizer(WordPiece(unk_token=UNK_TOKEN))

    # 2. 设置 Pre-tokenizer
    # Pre-tokenizer 负责将文本预先分割成“单词”。我们按空格分割。
    tokenizer.pre_tokenizer = Whitespace()

    # 3. 设置 Trainer
    # Trainer 负责从预分割的单词中学习和构建词表
    trainer = WordPieceTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS
    )

    # 4. 开始训练
    tokenizer.train_from_iterator(dataset_iterator, trainer=trainer)

    # 5. 保存分词器
    # 创建一个文件夹来存放它们
    save_dir = "tokenizers"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"tokenizer_{lang}.json")
    tokenizer.save(save_path)

    print(f"'{lang}' 分词器训练完毕并保存至: {save_path}")
    return tokenizer

# --- 4. 主执行函数 ---

if __name__ == "__main__":
    print("开始加载数据集...")
    # 1. 加载数据集
    dataset = load_dataset("bentrevett/multi30k")

    # 2. 训练英语 ('en') 分词器
    en_iterator = get_batch_iterator(dataset, 'en')
    en_tokenizer = train_and_save_tokenizer('en', en_iterator)

    # 3. 训练德语 ('de') 分词器
    de_iterator = get_batch_iterator(dataset, 'de')
    de_tokenizer = train_and_save_tokenizer('de', de_iterator)

    print("\n分词器训练完成！")

    # --- 5. 测试分词器 (可选) ---
    print("\n--- 测试 'en' 分词器 ---")
    text = "A boy in a red hat holding on to some railings."
    encoding = en_tokenizer.encode(text)
    print(f"原文: {text}")
    print(f"词元 (Tokens): {encoding.tokens}")
    print(f"ID (IDs): {encoding.ids}")

    print("\n--- 测试 'de' 分词器 ---")
    text_de = "Wännliches Kleinkind in einem roten Hut, das sich an einem Geländer festhält."
    encoding_de = de_tokenizer.encode(text_de)
    print(f"原文: {text_de}")
    print(f"词元 (Tokens): {encoding_de.tokens}")
    print(f"ID (IDs): {encoding_de.ids}")