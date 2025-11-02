## 数据集说明

根据作业要求“在代码仓库中提供数据集压缩包”，本文件夹用于展示数据。

* `samples_en_de.txt` 文件包含少量数据样本，用于直观地展示数据格式。

* 本项目使用的完整 `bentrevett/multi30k` 数据集，将在运行 `scripts/run.sh` (或 `src/build_tokenizer.py`) 时，通过 Hugging Face `datasets` 库自动从云端下载并缓存在用户本地。

这种方式既满足了可复现性的要求，也避免了将大型数据文件存储在 Git 仓库中。