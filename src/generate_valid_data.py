import torch
import random

# 固定随机种子
random.seed(42)
torch.manual_seed(42)

# 准备基础双语句子（英语→德语，共100个模板，扩展到1000个样本）
english_sentences = [
    "I love reading books",
    "She likes playing the piano",
    "We will go to the park tomorrow",
    "He is studying computer science",
    "They watched a movie last night",
    "The cat is sleeping on the sofa",
    "This restaurant serves delicious food",
    "I need to finish my homework",
    "She bought a new dress yesterday",
    "We enjoyed the concert very much",
    "He speaks English and German fluently",
    "The weather is beautiful today",
    "I want to travel around the world",
    "They are building a new house",
    "She teaches mathematics at school",
    "We drank coffee this morning",
    "He visited his grandparents last weekend",
    "The children are playing in the garden",
    "I have read this book twice",
    "She will prepare dinner for us"
]

german_sentences = [
    "Ich liebe es, Bücher zu lesen",
    "Sie mag Klavier spielen",
    "Wir gehen morgen ins Park",
    "Er studiert Informatik",
    "Sie haben gestern einen Film gesehen",
    "Die Katze schläft auf dem Sofa",
    "Dieses Restaurant serviert leckeres Essen",
    "Ich muss meine Hausaufgaben fertig machen",
    "Sie hat gestern ein neues Kleid gekauft",
    "Wir haben das Konzert sehr genossen",
    "Er spricht fließend Englisch und Deutsch",
    "Das Wetter ist heute wunderschön",
    "Ich möchte die Welt bereisen",
    "Sie bauen ein neues Haus",
    "Sie unterrichtet Mathematik in der Schule",
    "Wir haben heute Morgen Kaffee getrunken",
    "Er hat letztes Wochenende seine Großeltern besucht",
    "Die Kinder spielen im Garten",
    "Ich habe dieses Buch zweimal gelesen",
    "Sie wird für uns Abendessen zubereiten"
]

# 扩展到1000个样本（重复模板+轻微变体，保证数据量）
extended_english = []
extended_german = []
for _ in range(50):  # 每个模板重复5次，共20*50=1000样本
    for en, de in zip(english_sentences, german_sentences):
        # 轻微变体（避免完全重复，提升泛化）
        if random.random() > 0.7:
            en = en.replace("I", "I really") if "I " in en else en
            de = de.replace("Ich", "Ich wirklich") if "Ich " in de else de
        extended_english.append(en)
        extended_german.append(de)

# 划分训练集（900样本）、验证集（100样本）
train_size = 900
train_en = extended_english[:train_size]
train_de = extended_german[:train_size]
val_en = extended_english[train_size:]
val_de = extended_german[train_size:]

# 简单Tokenization（用空格分割，转换为索引）
# 构建简单词汇表（包含所有单词+特殊符号）
all_en_words = set(word for sent in extended_english for word in sent.split())
all_de_words = set(word for sent in extended_german for word in sent.split())

# 词汇表：<pad>=0, <unk>=1, <sos>=2, <eos>=3
src_vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
for idx, word in enumerate(all_en_words, 4):
    src_vocab[word] = idx

tgt_vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
for idx, word in enumerate(all_de_words, 4):
    tgt_vocab[word] = idx

# 保存词汇表（替换你原来的词汇表文件）
src_vocab_path = "../data/processed/src_vocab.txt"
tgt_vocab_path = "../data/processed/tgt_vocab.txt"
with open(src_vocab_path, "w", encoding="utf-8") as f:
    for word, idx in src_vocab.items():
        f.write(f"{word}\t{idx}\n")

with open(tgt_vocab_path, "w", encoding="utf-8") as f:
    for word, idx in tgt_vocab.items():
        f.write(f"{word}\t{idx}\n")

# 句子转换为索引（添加<sos>和<eos>，Padding到10个词）
def sentence_to_indices(sentence, vocab, max_len=10):
    tokens = sentence.split()
    # 转换为索引（未知词用<unk>）
    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    # 添加<sos>（开头）和<eos>（结尾）
    indices = [vocab["<sos>"]] + indices + [vocab["<eos>"]]
    # Padding到max_len
    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]  # 截断
    return torch.tensor(indices, dtype=torch.long)

# 生成训练集和验证集的Tensor格式
train_data = []
for en_sent, de_sent in zip(train_en, train_de):
    en_indices = sentence_to_indices(en_sent, src_vocab)
    de_indices = sentence_to_indices(de_sent, tgt_vocab)
    train_data.append( (en_indices, de_indices) )

val_data = []
for en_sent, de_sent in zip(val_en, val_de):
    en_indices = sentence_to_indices(en_sent, src_vocab)
    de_indices = sentence_to_indices(de_sent, tgt_vocab)
    val_data.append( (en_indices, de_indices) )

# 保存数据集（覆盖你原来的文件，确保路径正确）
train_data_path = "../data/processed/train_en-de.pt"
val_data_path = "../data/processed/validation_en-de.pt"
torch.save(train_data, train_data_path)
torch.save(val_data, val_data_path)

print(f"有效数据集生成完成！")
print(f"训练集：{len(train_data)}样本（保存至{train_data_path}）")
print(f"验证集：{len(val_data)}样本（保存至{val_data_path}）")
print(f"英语词汇表大小：{len(src_vocab)}（保存至{src_vocab_path}）")
print(f"德语词汇表大小：{len(tgt_vocab)}（保存至{src_vocab_path}）")