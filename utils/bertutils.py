from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert import BertTokenizer


def get_bert_optimizer(named_params, learning_rate, w_decay, extra_no_decay_keywords):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    for w in extra_no_decay_keywords:
        no_decay.append(w)

    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': w_decay},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate, warmup=bert_adam_warmup,
    #                      t_total=n_steps)
    return BertAdam(optimizer_grouped_parameters, lr=learning_rate)


def get_sample_bert_token_id_seq(bert_tokenizer: BertTokenizer, left_seq_str, right_seq_str, max_seq_len):
    left_bert_token_seq = bert_tokenizer.tokenize(left_seq_str)
    right_bert_token_seq = bert_tokenizer.tokenize(right_seq_str)

    if len(right_bert_token_seq) + 3 > max_seq_len:
        right_bert_token_seq = right_bert_token_seq[:max_seq_len - 3]

    if len(right_bert_token_seq) + len(left_bert_token_seq) + 3 > max_seq_len:
        left_bert_token_seq = left_bert_token_seq[:max_seq_len - len(right_bert_token_seq) - 3]

    bert_token_seq = ['[CLS]'] + left_bert_token_seq + ['[SEP]'] + right_bert_token_seq + ['[SEP]']
    # print(bert_token_seq)
    bert_token_id_seq = bert_tokenizer.convert_tokens_to_ids(bert_token_seq)
    return bert_token_id_seq
