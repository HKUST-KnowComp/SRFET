import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.mlp import MLP
from utils import fetutils
from models import modelutils


class SRLFET(nn.Module):
    def __init__(self, device, type_vocab, type_id_dict, word_vec_dim, lstm_dim, mlp_hidden_dim, type_embed_dim):
        super(SRLFET, self).__init__()

        # TODO to other class
        self.type_vocab, self.type_id_dict = type_vocab, type_id_dict
        self.l1_type_indices, self.l1_type_vec, self.child_type_vecs = fetutils.build_hierarchy_vecs(
            self.type_vocab, self.type_id_dict)
        self.n_types = len(self.type_vocab)

        self.device = device
        self.word_vec_dim = word_vec_dim
        self.lstm_dim = lstm_dim

        self.lstm1 = nn.LSTM(input_size=self.word_vec_dim, hidden_size=self.lstm_dim, bidirectional=False)
        self.lstm_hidden1 = None
        self.lstm2 = nn.LSTM(input_size=self.word_vec_dim, hidden_size=self.lstm_dim, bidirectional=False)
        self.lstm_hidden2 = None

        self.type_embed_dim = type_embed_dim
        self.type_embeddings = torch.tensor(np.random.normal(
            scale=0.01, size=(type_embed_dim, self.n_types)).astype(np.float32),
                                            device=self.device, requires_grad=True)
        self.type_embeddings = nn.Parameter(self.type_embeddings)

        self.mlp = MLP(2, 2 * self.word_vec_dim + 2 * self.lstm_dim, self.type_embed_dim, mlp_hidden_dim)
        # self.mlp = MLP(2, self.word_vec_dim, mlp_hidden_dim, type_embed_dim)

    def get_lstm_output(self, word_vec_seqs, lens, batch_size, arg_idx):
        lstm_model = self.lstm1 if arg_idx == 1 else self.lstm2
        # lstm_hidden = self.lstm_hidden1 if mention_arg_idx == 1 else self.lstm_hidden2

        lstm_hidden = modelutils.init_lstm_hidden(self.device, batch_size, self.lstm_dim, False)

        # x = F.dropout(x, self.dropout, training)
        x = torch.nn.utils.rnn.pack_padded_sequence(word_vec_seqs, lens, batch_first=True)
        lstm_output, lstm_hidden = lstm_model(x, lstm_hidden)

        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        return lstm_output

    def __get_arg_rep(self, arg_vec_seqs, arg_idx, batch_size):
        arg_vec_seqs, arg_seq_lens, arg_back_idxs = modelutils.get_len_sorted_vec_seqs_input(
            self.device, arg_vec_seqs)
        arg_lstm_output = self.get_lstm_output(arg_vec_seqs, arg_seq_lens, batch_size, arg_idx)
        arg_lstm_output = arg_lstm_output[:, -1, :]
        arg_lstm_output = arg_lstm_output[arg_back_idxs]
        return arg_lstm_output

    def forward(self, mstr_vec_seqs, verb_vec_seqs, arg1_vec_seqs, arg2_vec_seqs):
        batch_size = len(verb_vec_seqs)
        # arg1_vec_seqs, arg1_seq_lens, arg1_back_idxs = modelutils.get_len_sorted_vec_seqs_input(
        #     self.device, arg1_vec_seq)

        mstr_reps = modelutils.avg_vec_seqs(self.device, mstr_vec_seqs)
        verb_reps = modelutils.avg_vec_seqs(self.device, verb_vec_seqs)

        # arg1_vec_seqs, arg1_seq_lens, arg1_back_idxs = modelutils.get_len_sorted_vec_seqs_input(
        #     self.device, arg1_vec_seqs)
        # arg1_lstm_output = self.get_lstm_output(arg1_vec_seqs, arg1_seq_lens, batch_size, 1)
        # arg1_lstm_output = arg1_lstm_output[:, -1, :]
        # arg1_lstm_output = arg1_lstm_output[arg1_back_idxs]
        arg1_lstm_output = self.__get_arg_rep(arg1_vec_seqs, 1, batch_size)
        arg2_lstm_output = self.__get_arg_rep(arg2_vec_seqs, 2, batch_size)

        reps = torch.cat((mstr_reps, verb_reps, arg1_lstm_output, arg2_lstm_output), dim=1)
        final_reps = self.mlp(reps)

        logits = torch.matmul(final_reps.view(-1, 1, self.type_embed_dim),
                              self.type_embeddings.view(-1, self.type_embed_dim, self.n_types))
        logits = logits.view(-1, self.n_types)
        return logits

    def get_loss(self, true_type_vecs, scores, margin=1.0, person_loss_vec=None):
        tmp1 = torch.sum(true_type_vecs * F.relu(1.0 - scores), dim=1)
        # tmp2 = torch.sum((1 - true_type_vecs) * F.relu(margin + scores), dim=1)
        tmp2 = (1 - true_type_vecs) * F.relu(1.0 + scores)
        if person_loss_vec is not None:
            tmp2 *= person_loss_vec.view(-1, self.n_types)
        tmp2 = torch.sum(tmp2, dim=1)
        loss = torch.mean(torch.add(tmp1, tmp2))
        return loss

    def get_cali_loss(self, true_type_vecs, scores):
        tmp1, _ = torch.max(true_type_vecs * F.relu(1.0 - scores), dim=1)
        tmp2, _ = torch.max((1 - true_type_vecs) * F.relu(1.0 + scores), dim=1)
        loss = torch.mean(torch.add(tmp1, tmp2))
        return loss

    def inference(self, scores, is_torch_tensor=True):
        if is_torch_tensor:
            scores = scores.data.cpu().numpy()
        return fetutils.inference_labels(self.l1_type_indices, self.child_type_vecs, scores)

    def inference_full(self, logits, extra_label_thres=0.5, is_torch_tensor=True):
        if is_torch_tensor:
            logits = logits.data.cpu().numpy()
        return fetutils.inference_labels_full(self.l1_type_indices, self.child_type_vecs, logits, extra_label_thres)
