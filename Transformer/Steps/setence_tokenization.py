from torch.utils.data import Dataset, DataLoader
from positional_encoding import PositionalEncoding
import torch
import numpy as np
import torch.nn as nn
import time

start_time = time.time()


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


english_file = '/Users/gunneo/Codes/Transformer/Steps/english.txt'
kannada_file = '/Users/gunneo/Codes/Transformer/Steps/kannada.txt'

START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

kannada_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'",
                      '(', ')', '*', '+', ',', '-', '.', '/',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':',
                      '<', '=', '>', '?', 'ˌ',
                      'ँ', 'ఆ', 'ఇ', 'ా', 'ి', 'ీ', 'ు', 'ూ',
                      'ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ೠ', 'ಌ', 'ಎ',
                      'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ',
                      'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ',
                      'ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ',
                      'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ',
                      'ತ', 'ಥ', 'ದ', 'ಧ', 'ನ',
                      'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ',
                      'ಯ', 'ರ', 'ಱ', 'ಲ', 'ಳ', 'ವ', 'ಶ', 'ಷ', 'ಸ', 'ಹ',
                      '಼', 'ಽ', 'ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೃ', 'ೄ', 'ೆ', 'ೇ',
                      'ೈ', 'ೊ', 'ೋ', 'ೌ', '್', 'ೕ', 'ೖ', 'ೞ', 'ೣ', 'ಂ', 'ಃ',
                      '೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯',
                      PADDING_TOKEN, END_TOKEN]
english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'",
                      '(', ')', '*', '+', ',', '-', '.', '/',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      ':', '<', '=', '>', '?', '@',
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                      'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                      'W', 'X', 'Y', 'Z',
                      '[', '\\', ']', '^', '_', '`',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                      'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                      'w', 'x', 'y', 'z',
                      '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

index_to_kannada = {k: v for k, v in enumerate(kannada_vocabulary)}
kannada_to_index = {v: k for k, v in enumerate(kannada_vocabulary)}
index_to_english = {k: v for k, v in enumerate(english_vocabulary)}
english_to_index = {v: k for k, v in enumerate(english_vocabulary)}

with open(english_file, 'r') as file:
    english_sentences = file.readlines()
with open(kannada_file, 'r') as file:
    kannada_sentences = file.readlines()

# Limit Number of sentences
TOTAL_SENTENCES = 100000
english_sentences = english_sentences[:TOTAL_SENTENCES]
kannada_sentences = kannada_sentences[:TOTAL_SENTENCES]
english_sentences = [sentence.rstrip('\n') for sentence in english_sentences]
kannada_sentences = [sentence.rstrip('\n') for sentence in kannada_sentences]

PERCENTILE = 97
english_precentile = np.percentile([len(x) for x in english_sentences],
                                   PERCENTILE)  # 179
kannada_precentile = np.percentile([len(x) for x in english_sentences],
                                   PERCENTILE)  # 172

max_sequence_length = 200


def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True


def is_valid_length(sentence, max_sequence_length):
    # need to re-add the end token so leaving 1 space
    return len(list(sentence)) < (max_sequence_length - 1)


valid_sentence_indicies = []
for index in range(len(kannada_sentences)):
    kannada_sentence = kannada_sentences[index]
    english_sentence = english_sentences[index]
    if is_valid_length(kannada_sentence, max_sequence_length) \
            and is_valid_length(english_sentence, max_sequence_length) \
            and is_valid_tokens(kannada_sentence, kannada_vocabulary) \
            and is_valid_tokens(english_sentence, english_vocabulary):
        valid_sentence_indicies.append(index)

kannada_sentences = [kannada_sentences[i] for i in valid_sentence_indicies]
english_sentences = [english_sentences[i] for i in valid_sentence_indicies]


class TextDataset(Dataset):
    def __init__(self, english_sentences, kannada_sentences):
        self.english_sentences = english_sentences
        self.kannada_sentences = kannada_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.kannada_sentences[idx]


dataset = TextDataset(english_sentences, kannada_sentences)
batch_size = 3  # 3 sentences will be simultaneously trained
train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)

for batch_num, batch in enumerate(iterator):
    if batch_num > 3:
        break


def tokenize(sentence, language_to_index, start_token=True, end_token=True):
    sentence_word_indicies = [language_to_index[token]
                              for token in list(sentence)]
    if start_token:
        sentence_word_indicies.insert(0, language_to_index[START_TOKEN])
    if end_token:
        sentence_word_indicies.append(language_to_index[END_TOKEN])
    for _ in range(len(sentence_word_indicies), max_sequence_length):
        sentence_word_indicies.append(language_to_index[PADDING_TOKEN])
    return torch.tensor(sentence_word_indicies)


eng_tokenized, kn_tokenized = [], []
for sentence_num in range(batch_size):
    eng_sentence, kn_sentence = batch[0][sentence_num], batch[1][sentence_num]
    eng_tokenized.append(
        tokenize(eng_sentence, english_to_index,
                 start_token=False, end_token=False))
    kn_tokenized.append(
        tokenize(kn_sentence, kannada_to_index,
                 start_token=True, end_token=True))
eng_tokenized = torch.stack(eng_tokenized)
kn_tokenized = torch.stack(kn_tokenized)

# print(len(english_vocabulary), len(kannada_vocabulary))
# print(kn_tokenized[0])

NEG_INFTY = -1e9


def create_masks(eng_batch, kn_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full(
        [max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full(
        [num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attention = torch.full(
        [num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attention = torch.full(
        [num_sentences, max_sequence_length, max_sequence_length], False)

    for idx in range(num_sentences):
        eng_sentence_length, kn_sentence_length = len(
            eng_batch[idx]), len(kn_batch[idx])
        # create the mask with padding in encoder
        eng_chars_to_padding_mask = np.arange(
            eng_sentence_length + 1, max_sequence_length)
        # create the mask with padding in decoder
        kn_chars_to_padding_mask = np.arange(
            kn_sentence_length + 1, max_sequence_length)
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx,
                                            :, kn_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx,
                                            kn_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :,
                                             eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask,
                                             :] = True

    # change true(padding) into -inf and false to 0
    encoder_self_attention_mask = torch.where(
        encoder_padding_mask, NEG_INFTY, 0)
    # tensor broadcasting technique
    decoder_self_attention_mask = torch.where(
        look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(
        decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask,
    decoder_cross_attention_mask


create_masks(batch[0], batch[1])


class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"

    def __init__(self, max_sequence_length, d_model, language_to_index,
                 START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(
            d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

    def batch_tokenize(self, batch, start_token=True, end_token=True):

        def tokenize(sentence, start_token=True, end_token=True):
            sentence_word_indicies = [
                self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(
                    0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(
                    self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies),
                           self.max_sequence_length):
                sentence_word_indicies.append(
                    self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append(
                tokenize(batch[sentence_num], start_token, end_token))
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())

    def forward(self, x, end_token=True):  # sentence
        x = self.batch_tokenize(x, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x


end_time = time.time()

print(f"Elapsed Time: {end_time - start_time} seconds")
