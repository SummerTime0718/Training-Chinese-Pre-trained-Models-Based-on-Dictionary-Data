from tokenizer import Tokenizer
import sentencepiece as spm
import os

data_file = './data/interpre.txt'
prefix = './data/tok14000'
vocab_size =14000
print("開始訓練")
spm.SentencePieceTrainer.train(input=data_file,
                                   model_prefix= prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size,
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=1.0,
                                   num_threads=os.cpu_count(),
                                   split_digits=True,
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=r" \342\201\207 ",
                                   normalization_rule_name="identity",
                                   max_sentence_length=5000
                                   )
print("訓練完成")