import jieba

def tokenize(text: str) -> list[str]:
    """共用的中文分词函数，确保训练和推理时序列化路径一致。"""
    return jieba.lcut(text)