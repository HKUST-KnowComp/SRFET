import torch


def init_lstm_hidden(device, batch_size, hidden_dim, bidirectional):
    d = 2 if bidirectional else 1
    return (torch.zeros(d, batch_size, hidden_dim, requires_grad=True, device=device),
            torch.zeros(d, batch_size, hidden_dim, requires_grad=True, device=device))


def get_vec_seqs_torch_input(device, vec_seqs):
    seq_lens = torch.tensor([len(seq) for seq in vec_seqs], dtype=torch.long, device=device)
    vec_seqs = torch.nn.utils.rnn.pad_sequence(vec_seqs, batch_first=True)
    return vec_seqs, seq_lens


def get_len_sorted_vec_seqs_input(device, vec_seqs):
    data_tups = list(enumerate(vec_seqs))
    data_tups.sort(key=lambda x: -len(x[1]))
    vec_seqs = [x[1] for x in data_tups]
    idxs = [x[0] for x in data_tups]
    back_idxs = [0] * len(idxs)
    for i, idx in enumerate(idxs):
        back_idxs[idx] = i

    back_idxs = torch.tensor(back_idxs, dtype=torch.long, device=device)
    seqs, seq_lens = get_vec_seqs_torch_input(device, vec_seqs)

    return seqs, seq_lens, back_idxs


def avg_vec_seqs(device, vec_seqs):
    lens = torch.tensor([len(seq) for seq in vec_seqs], dtype=torch.float32, device=device
                        ).view(-1, 1)
    vec_seqs = torch.nn.utils.rnn.pad_sequence(vec_seqs, batch_first=True)
    vecs_avg = torch.div(torch.sum(vec_seqs, dim=1), lens)
    return vecs_avg
