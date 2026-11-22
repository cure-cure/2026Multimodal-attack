import jieba


def tokenize(text: str) -> list[str]:
    """共用的中文分词函数，确保训练和推理时序列化路径一致。

    ⚠️ 这个函数被 v2 模型的 pickle 按名字引用（`TfidfVectorizer.tokenizer=tokenize`），
    改它的行为会让已发布的 v2 模型分数静默漂移。要换分词策略请新增函数，不要改这里。
    """
    return jieba.lcut(text)


def tokenize_drop_space(text: str) -> list[str]:
    """v3 用：丢掉纯空白 token。

    jieba 对英文会吐出 ' ' 这类纯空格 token，TF-IDF 的 (1,2)-gram 于是把
    'How' + ' ' 拼成 'How  ' 这种噪声特征，把真正的词组合淹掉。
    去掉之后英文侧的 bigram 才是 'How can' 这种有意义的组合。
    """
    return [tok for tok in jieba.lcut(text) if tok.strip()]
