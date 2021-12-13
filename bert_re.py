import json
from pprint import pprint
import sys
import os
import re
import argparse
from os.path import join
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import init
import numpy as np

from transformers import BertTokenizerFast, AutoConfig, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup

here = os.path.dirname(os.path.abspath(__file__))

default_pretrained_model_path = os.path.join(here, '/home/juan.du/nlp/bert_models/bert-base-chinese')
default_data_path = os.path.join(here,'data')
default_train_file = os.path.join(default_data_path, 'train_data.json')
default_validation_file = os.path.join(here, './data/dev_data.json')
default_output_dir = os.path.join(here, './saved_models')
default_log_dir = os.path.join(default_output_dir, 'runs')
default_tagset_file = os.path.join(default_data_path, 'relation.txt')
default_model_file = os.path.join(default_output_dir, 'model.bin')
default_checkpoint_file = os.path.join(default_output_dir, 'checkpoint.json')

parser = argparse.ArgumentParser(description='PyTorch bert_re Sequence Labeling')

parser.add_argument('--model-path', type=str, default=default_output_dir, metavar='S',
                    help='model path')
parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training')
parser.add_argument('--dev-batch-size', type=int, default=64, metavar='N',
                    help='dev batch size')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--embedding-size', type=int, default=256, metavar='N',
                    help='embedding size')
parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                    help='hidden size')
parser.add_argument('--rnn-layer', type=int, default=1, metavar='N',
                    help='RNN layer num')
parser.add_argument('--with-layer-norm', action='store_true', default=False,
                    help='whether to add layer norm after RNN')
parser.add_argument('--dropout', type=float, default=0, metavar='RATE',
                    help='dropout rate')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed')
parser.add_argument('--save-interval', type=int, default=30, metavar='N',
                    help='save interval')
parser.add_argument('--valid-interval', type=int, default=60, metavar='N',
                    help='valid interval')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='log interval')
parser.add_argument('--patience', type=int, default=30, metavar='N',
                    help='patience for early stop')
parser.add_argument('--vocab', nargs='+', required=False, metavar='SRC_VOCAB TGT_VOCAB',
                    help='src vocab and tgt vocab')
parser.add_argument('--trainset', type=str, default=default_train_file, metavar='trainset',
                    help='trainset path')
parser.add_argument('--devset', type=str, default=default_validation_file, metavar='devset',
                    help='devset path')
parser.add_argument("--pretrained_model_path", type=str, default=default_pretrained_model_path)

args = parser.parse_args()

START_TAG = "<start>"
END_TAG = "<end>"
PAD = "<pad>"
UNK = "<unk>"

def find_all_span(text,span_text):
    last_end = 0
    tmp_lst = []
    if len(text) * len(span_text) < 1:
        return tmp_lst 
    while True:
        istart = text[last_end:].find(span_text)
        if istart > -1 and last_end < len(text) - 1:
            iend = len(span_text) + istart + 1 + last_end
            tmp_lst.append((istart+last_end,iend-1))
            last_end = iend - 1
        else:
            break
    return tmp_lst
def build_corpus(data_file, make_vocab=True):
    """读取数据"""
    if not make_vocab:
        with open(os.path.join(default_data_path,'rtag2id.json'),'r') as fr1:
            rtag2id = json.load(fr1)
        with open(os.path.join(default_data_path,'nertag2id.json'),'r') as fr2:
            nertag2id = json.load(fr2)
        
    texts, postags, nertag_lists, rlists = [],[],[],[]
    rset, posset = set(),set()
    ner_typ_set = {"O"}
    e_typ_set = set()

    with open(data_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            pos_list = []
            rlist = []
            data = json.loads(line)
            if len(data["postag"]) < 1: # 没有词性信息的去掉
                continue
            text = data["text"].lower()
            re_del = re.compile(r"[^\u4e00-\u9fa5A-Za-z0-9\(\)\[\]\{\}<>（）【】「」《》。，；、\.,;\-_·]")
            text = re.sub(re_del,"",text)
            text = text.strip()
            if len(text) < 1:
                continue
            # 词性
            for ipos_dict in data["postag"]:
                pos_word = ipos_dict["word"].lower()
                pos_word = re.sub(re_del,"",pos_word)
                pos_tag = ipos_dict["pos"]
                pos_list.extend([pos_tag]*len(pos_word))
                posset.add(pos_tag)
            for ispo_dict in data["spo_list"]: 
                obj = ispo_dict["object"].lower()
                obj = re.sub(re_del,"",obj)
                obj_typ = ispo_dict["object_type"]
                e_typ_set.add(obj_typ)
                
                subj = ispo_dict["subject"].lower()
                subj = re.sub(re_del,"",subj)
                subj_typ = ispo_dict["subject_type"]
                e_typ_set.add(subj_typ)
                        
                obj_span = find_all_span(text,obj)
                subj_span = find_all_span(text,subj)
                
                for i in obj_span:
                    if obj != text[i[0]:i[1]]:
                        print(obj)
                        print(text[i[0]:i[1]])
                        print(data["text"])
                        print(ispo_dict["object"])
                        sys.exit()
                for i in subj_span:
                    if subj != text[i[0]:i[1]]:
                        print(sub)
                        print(text[i[0]:i[1]])
                        print(data["text"])
                        print(ispo_dict["subject"])
                        sys.exit()                
                               
                r = ispo_dict["predicate"]
                rset.add(r)
                if len(obj_span)*len(subj_span) > 0:
                    rlist.append(((obj_span,obj_typ),(subj_span,subj_typ),r))
               
            nertag_list = ['O'] * len(text)
            for rtuple in rlist: 
                obj_span_lst, obj_typ = rtuple[0]
                ner_typ_set.add("B-" + obj_typ)  
                ner_typ_set.add("I-" + obj_typ)
                for j in obj_span_lst:
                    start,end = j[0],j[1]
                    nertag_list[start] = "B-" + obj_typ
                    if end > start + 1:
                        nertag_list[start+1:end] = ["I-" + obj_typ]*(end-start-1)
                     
                subj_span_lst, subj_typ = rtuple[1]
                ner_typ_set.add("B-" + subj_typ)  
                ner_typ_set.add("I-" + subj_typ)
                
                for k in subj_span_lst:
                    start,end = k[0],k[1]
                    nertag_list[start] = "B-" + subj_typ
                    if end > start + 1:
                        nertag_list[start+1:end] = ["I-" + subj_typ]*(end-start-1)

#             del_ind = {i for i,j in enumerate(list(text)) if j in [' ','\t','\u3000']}
#             pos_list = [j for i,j in enumerate(pos_list)]
#             nertag_list = [j for i,j in enumerate(nertag_list)]
#             text = "".join([j for i,j in enumerate(list(text))])
            try:
                assert len(text) == len(pos_list) == len(nertag_list)
            except:
                print("text: ",text, len(text))
                print("pos_list: ",pos_list, len(pos_list))
                print("nertag_list: ",nertag_list, len(nertag_list))
                continue
            postags.append(pos_list)
            nertag_lists.append(nertag_list)
            texts.append(text)
            rlists.append(rlist)
    # 如果make_vocab为True，还需要返回tag2id
    print("number of data: ",len(texts))
    if make_vocab:       
        pos2id = {PAD:0, UNK:1}
        pos2id.update({k:v+2 for v,k in enumerate(posset)})
        nertag2id = {PAD:0, START_TAG:1, END_TAG:2}
        nertag2id.update({k:v+3 for v,k in enumerate(ner_typ_set)})
        rtag2id = {k:v for v,k in enumerate(rset)}
        rtag2id['norelation'] = len(rtag2id) # 添加无关系分类
        with open(os.path.join(default_data_path,'rtag2id.json'),'w') as fw1:
            json.dump(rtag2id,fw1,ensure_ascii=False)
        with open(os.path.join(default_data_path,'nertag2id.json'),'w') as fw2:
            json.dump(nertag2id,fw2,ensure_ascii=False)
        return texts, nertag_lists, postags, rlists, pos2id, nertag2id, rtag2id, {k:v for v,k in enumerate(e_typ_set)}
    else:
        return texts, nertag_lists, postags, rlists

class REDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, texts, postags, nertag_lists, rlists, pos2id, nertag2id, rtag2id, tokenizer_path = '') :
        ## 一般init函数是加载所有数据
        super(REDataset, self).__init__()
        # 读原始数据
        # self.texts, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.texts = texts
        self.postags = postags
        self.nertag_lists = nertag_lists
        self.rlists = rlists
        self.pos2id = pos2id
        self.nertag2id = nertag2id
        self.rtag2id = rtag2id
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    def __getitem__(self, i):
        ## 得到单个数据
        src = self.texts[i]
        nertag = [self.nertag2id[j] for j in self.nertag_lists[i]] 
        postag = [self.pos2id[j] for j in self.postags[i]] # 不含cls,sep位置
        tokenized = self.tokenizer.encode_plus(src, return_offsets_mapping=True, add_special_tokens=True)
#         tokenized = tokenizer_fast([src], is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        # e.g.
        # {'input_ids': [101, 2769, 3221, 8604, 702, 1368, 2094, 102], 
        # 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0], 
        # 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1], 
        # 'offset_mapping': [(0, 0), (0, 1), (1, 2), (2, 5), (5, 6), (6, 7), (7, 8), (0, 0)]}
#         input_ids, token_type_ids, attention_mask = self.tokenizer(src)
        # 保证tokens和字数保持一致
        offset_map = [max(j[1]-j[0],1) for j in tokenized["offset_mapping"]]
        input_ids = np.repeat(np.array(tokenized["input_ids"]), offset_map).tolist()
        token_type_ids = np.repeat(np.array(tokenized["token_type_ids"]), offset_map).tolist()
        attention_mask = np.repeat(np.array(tokenized["attention_mask"]), offset_map).tolist()
        try:
          assert len(src) == len(input_ids)-2
        except:
          print("="*10)
          print(len(src),src)
          print(len(input_ids),input_ids)
          sys.exit()
        output = {
            "input_ids": input_ids,
            "postag_ids": postag,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "nertag_ids": nertag, #
            "rtag_ids": self.rlists[i]
        }
        
        
        tmp = input_ids[1:-1]
        
        for irpair in self.rlists[i]:
#             print("irpair:",irpair)
            if len(irpair[0][0]) > 1:
#                 print("irpair[0][0]: ",irpair[0][0])
                tmp_token_span = tmp[irpair[0][0][0][0]:irpair[0][0][0][1]]
            else:
                continue
            for ii in irpair[0][0]:
                if tmp[ii[0]:ii[1]] != tmp_token_span:
                    print("1*"*10)
                    print(self.rlists[i])
                    print(tmp[ii[0]:ii[1]])
                    print(ii)
                    print(tmp_token_span)
                    print(src[ii[0]:ii[1]])
                    print(src)
                    sys.exit()
#         print([tmp[ii:jj] for ii,jj in set(self.rlists[i][0][0])])
        
        return output

    def __len__(self):
        return len(self.texts) 
    def __offset_map(self, offset_mapping):
        return [max(i[1]-i[0],1) for i in offset_mapping]
def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.LongTensor(pad_indice)
    max_length = max([len(i["input_ids"]) for i in batch])
  
    token_ids_padded = padding([i["input_ids"] for i in batch], max_length, pad_idx=0)
    token_type_ids_padded = padding([i["token_type_ids"] for i in batch], max_length, pad_idx=0)
    att_mask_padded = padding([i["attention_mask"] for i in batch], max_length, pad_idx=0)
    target_ids_padded = padding([i["nertag_ids"] for i in batch], max_length - 2, pad_idx=nertag2id[PAD]).transpose(0,1) # 不包含cls,sep位置
    pos_ids_padded = padding([i["postag_ids"] for i in batch], max_length - 2, pad_idx=pos2id[PAD])
    crf_mask = att_mask_padded[:,2:].transpose(0,1)
    rtag_ids = [i["rtag_ids"] for i in batch]
    try:
        assert token_ids_padded.shape[1] == target_ids_padded.shape[0]+2 == pos_ids_padded.shape[1]+2
    except:
        print("token_ids_padded.shape,target_ids_padded.shape,pos_ids_padded.shape: ",token_ids_padded.shape,target_ids_padded.shape)
        sys.exit()
    return token_ids_padded, token_type_ids_padded, att_mask_padded, pos_ids_padded, target_ids_padded, crf_mask, rtag_ids
            
def log_sum_exp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    """
    Compute logsumexp in a numerically stable way.
    This is mathematically equivalent to ``tensor.exp().sum(dim, keep=keepdim).log()``.
    This function is typically used for summing log probabilities.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()

class CRFLayer(nn.Module):
  def __init__(self, tag_size):
    super(CRFLayer, self).__init__()
    # transition[i][j] means transition probability from j to i
    self.transition = nn.Parameter(torch.randn(tag_size, tag_size))

    self.reset_parameters()

  def reset_parameters(self):
    init.normal_(self.transition)
    # initialize START_TAG, END_TAG probability in log space
    self.transition.detach()[nertag2id[START_TAG], :] = -10000
    self.transition.detach()[:, nertag2id[END_TAG]] = -10000

  def forward(self, feats, mask):
    """
    Arg:
      feats: (seq_len, batch_size, tag_size)
      mask: (seq_len, batch_size)
    Return:
      scores: (batch_size, )
    """
    seq_len, batch_size, tag_size = feats.size()
    # initialize alpha to zero in log space
    alpha = feats.new_full((batch_size, tag_size), fill_value=-10000)
    # alpha in START_TAG is 1
    alpha[:, nertag2id[START_TAG]] = 0
    for t, feat in enumerate(feats):
      # broadcast dimension: (batch_size, next_tag, current_tag)
      # emit_score is the same regardless of current_tag, so we broadcast along current_tag
      emit_score = feat.unsqueeze(-1) # (batch_size, tag_size, 1)
      # transition_score is the same regardless of each sample, so we broadcast along batch_size dimension
      transition_score = self.transition.unsqueeze(0) # (1, tag_size, tag_size)
      # alpha_score is the same regardless of next_tag, so we broadcast along next_tag dimension
      alpha_score = alpha.unsqueeze(1) # (batch_size, 1, tag_size)
      alpha_score = alpha_score + transition_score + emit_score
      # log_sum_exp along current_tag dimension to get next_tag alpha
      mask_t = mask[t].unsqueeze(-1)
      alpha = log_sum_exp(alpha_score, -1) * mask_t + alpha * (1 - mask_t) # (batch_size, tag_size)
    # arrive at END_TAG
    alpha = alpha + self.transition[nertag2id[END_TAG]].unsqueeze(0)

    return log_sum_exp(alpha, -1) # (batch_size, )

  def score_sentence(self, feats, tags, mask):
    """
    Arg:
      feats: (seq_len, batch_size, tag_size)
      tags: (seq_len, batch_size)
      mask: (seq_len, batch_size)
    Return:
      scores: (batch_size, )
    """
    seq_len, batch_size, tag_size = feats.size()
    scores = feats.new_zeros(batch_size)
    tags = torch.cat([tags.new_full((1, batch_size), fill_value=nertag2id[START_TAG]), tags], 0) # (seq_len + 1, batch_size)
    for t, feat in enumerate(feats):
      emit_score = torch.stack([f[next_tag] for f, next_tag in zip(feat, tags[t + 1])])
      transition_score = torch.stack([self.transition[tags[t + 1, b], tags[t, b]] for b in range(batch_size)])
      scores += (emit_score + transition_score) * mask[t]
    transition_to_end = torch.stack([self.transition[nertag2id[END_TAG], tag[mask[:, b].sum().long()]] for b, tag in enumerate(tags.transpose(0, 1))])
    scores += transition_to_end
    return scores

  def viterbi_decode(self, feats, mask):
    """
    :param feats: (seq_len, batch_size, tag_size)
    :param mask: (seq_len, batch_size)
    :return best_path: (seq_len, batch_size)
    """
    seq_len, batch_size, tag_size = feats.size()
    # initialize scores in log space
    scores = feats.new_full((batch_size, tag_size), fill_value=-10000)
    scores[:, nertag2id[START_TAG]] = 0
    pointers = []
    # forward
    for t, feat in enumerate(feats):
      # broadcast dimension: (batch_size, next_tag, current_tag)
      scores_t = scores.unsqueeze(1) + self.transition.unsqueeze(0)  # (batch_size, tag_size, tag_size)
      # max along current_tag to obtain: next_tag score, current_tag pointer
      scores_t, pointer = torch.max(scores_t, -1)  # (batch_size, tag_size), (batch_size, tag_size)
      scores_t += feat
      pointers.append(pointer)
      mask_t = mask[t].unsqueeze(-1)  # (batch_size, 1)
      scores = scores_t * mask_t + scores * (1 - mask_t)
    pointers = torch.stack(pointers, 0) # (seq_len, batch_size, tag_size)
    scores += self.transition[nertag2id[END_TAG]].unsqueeze(0)
    best_score, best_tag = torch.max(scores, -1)  # (batch_size, ), (batch_size, )
    # backtracking
    best_path = best_tag.unsqueeze(-1).tolist() # list shape (batch_size, 1)
    for i in range(batch_size):
      best_tag_i = best_tag[i]
      seq_len_i = int(mask[:, i].sum())
      for ptr_t in reversed(pointers[:seq_len_i, i]):
        # ptr_t shape (tag_size, )
        best_tag_i = ptr_t[best_tag_i].item()
        best_path[i].append(best_tag_i)
      # pop first tag
      best_path[i].pop()
      # reverse order
      best_path[i].reverse()
    return best_path
class BERT_CRF_NER(nn.Module):
  def __init__(self, bert_model_path, config, tag_size, pos_size):
    super(BERT_CRF_NER, self).__init__()
#     self.dropout = nn.Dropout(dropout)
    self.bert = AutoModel.from_pretrained(bert_model_path)
    self.pos_emb_size = 100
    self.posemb = nn.Embedding(pos_size, self.pos_emb_size, padding_idx=pos2id[PAD])
    self.bertlinear = nn.Linear(config.hidden_size, config.hidden_size-self.pos_emb_size)
    self.hidden2tag = nn.Linear(config.hidden_size, tag_size)
    self.crf = CRFLayer(tag_size)
    self.reset_parameters()

  def reset_parameters(self):
    init.xavier_normal_(self.hidden2tag.weight)
    init.xavier_normal_(self.posemb.weight)
    init.xavier_normal_(self.bertlinear.weight)

  def get_bert_features(self, token_ids, token_type_ids, att_mask, pos_ids):
    bert_outputs = self.bert(token_ids, token_type_ids, att_mask).last_hidden_state[:,:-1,:] # (batch_size, seq_len, embedding_size)  
    bert_tokens = bert_outputs[:,1:,:]
    pos_emb = self.posemb(pos_ids)
    bert_emb = self.bertlinear(bert_tokens)
    enhanced_features = torch.cat([pos_emb,bert_emb],dim=-1)
    out_features = self.hidden2tag(enhanced_features).transpose(0,1)   # (seq_len, batch_size, tag_size)
    return out_features 
    
  def get_bert_encodings(self, token_ids, token_type_ids, att_mask):
    output = self.bert(token_ids, token_type_ids, att_mask).last_hidden_state
    pooled_output = output[:,0,:] 
    seq_output = output[:,1:-1,:]
    return seq_output, pooled_output
    
  def neg_log_likelihood(self, token_ids, token_type_ids, att_mask, pos_ids, target_ids, crf_mask):
    """
    :param tags: (seq_len, batch_size)
    :param mask: (seq_len, batch_size)
    # tags: (seq_len, batch_size)
    """ 
    bert_features = self.get_bert_features(token_ids, token_type_ids, att_mask, pos_ids)
    forward_score = self.crf(bert_features, crf_mask) # mask: (seq_len, batch_size)
    gold_score = self.crf.score_sentence(bert_features, target_ids, crf_mask)
    loss = (forward_score - gold_score).sum()

    return loss

  def predict(self, token_ids, token_type_ids, att_mask, pos_ids, crf_mask):
    """
    :param mask: (seq_len, batch_size)
    """
    bert_features = self.get_bert_features(token_ids, token_type_ids, att_mask, pos_ids)
    best_paths = self.crf.viterbi_decode(bert_features, crf_mask)

    return best_paths

def get_entity_with_entity_num(tags,tokens):
  tokens = tokens.tolist()[1:-1]
  entity = defaultdict(list)
  prev_entity = "O"
  start = -1
  end = -1
  mask_dict = defaultdict(list) # {entity_num1:mask1,entity_num2:mask2...}
  entities_dict = dict() # ['[token_span]': entity 编号] # 不同的token_span可以对应相同的实体编号（是同一个实体的不同mention）
  for i, tag in enumerate(tags):
    if tag[0] == "O":
      if prev_entity != "O":
        itokens = str(tokens[start:end+1])
        if itokens not in list(entities_dict):
          entity_num = len(entities_dict)
          entities_dict.update({itokens:entity_num})
        else:
          entity_num = entities_dict[itokens]
        entity[entity_num].append((prev_entity, start, end+1))
      prev_entity = "O"
    if tag[0] == "B":
      if prev_entity != "O":
        itokens = str(tokens[start:end+1])
        if itokens not in list(entities_dict):
          entity_num = len(entities_dict)
          entities_dict.update({itokens:entity_num})
        else:
          entity_num = entities_dict[itokens]
        entity[entity_num].append((prev_entity, start, end+1))
      prev_entity = tag[2:]
      start = end = i
    if tag[0] in ["M","E","I"]:
      if prev_entity == tag[2:]:
        end = i
      if i == len(tags) - 1:
        itokens = str(tokens[start:end+1])
        if itokens not in list(entities_dict):
          entity_num = len(entities_dict)
          entities_dict.update({itokens:entity_num})
        else:
          entity_num = entities_dict[itokens]
        entity[entity_num].append((prev_entity, start, end+1))
  for ith_entity, span_lst in entity.items():
    if ith_entity not in list(mask_dict):
      mask_dict[ith_entity] = len(tokens)*[0]
    for _, i, j in span_lst:
      mask_dict[ith_entity][i:j] = [1]*(j-i)
  return entity, mask_dict 
class SentenceRE(nn.Module):
    def __init__(self, bert_model_path, bert_config, ner_tag_size, pos_size, etyp2id):
        super(SentenceRE, self).__init__()
         
        self.bert_ner = BERT_CRF_NER(bert_model_path, bert_config, ner_tag_size, pos_size)
        self.embedding_dim = bert_config.hidden_size
        self.dropout = 0.1
        self.rtag_size = len(rtag2id)
        self.etyp_size = len(etyp2id)
        self.etyp2id = etyp2id
        
        self.etype_emb = nn.Embedding(self.etyp_size, self.embedding_dim)
        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.drop = nn.Dropout(self.dropout)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(self.embedding_dim * 5)
        self.classifier = nn.Linear(self.embedding_dim * 5, self.rtag_size)
        
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
#         self.loss_weight = torch.nn.parameter.Parameter(torch.FloatTensor(1))
#         self.loss_weight = nn.Parameter(torch.tensor([1.0],requires_grad=True))

    def cal_loss(self, token_ids, token_type_ids, att_mask, pos_ids, crf_mask, nertag_ids, rlists):
        best_paths = self.bert_ner.predict(token_ids, token_type_ids, att_mask, pos_ids, crf_mask)
        sequence_output, pooled_output = self.bert_ner.get_bert_encodings(token_ids, token_type_ids, att_mask)
        rtag_ids = []
        cnt_cyc = 0
        bs = len(best_paths)
        logits_expanded_e_dict = defaultdict(list) #
        for idx in range(bs): # iterate in one batchs, idx-th instance in one batch
            entities, mask_dict = get_entity_with_entity_num([id2nertag[i] for i in best_paths[idx]], token_ids[idx])
            # 只考虑存在关系的实体
            true_e_lst = list()
            for i in rlists[idx]:
                true_e_lst.extend(i[0][0] + i[1][0])
            for i_e, espan in entities.items():
                if espan[0][1:3] not in true_e_lst: # 只要验证第一个mention span是否在其中
                    mask_dict.pop(i_e)
#             print("true_e_lst")
#             print(true_e_lst)
#             sys.exit()
            for ith_e, jth_e in combinations(list(mask_dict), 2): # expand in one instance，遍历一个实例中所有实体对。
                e1_mask = torch.LongTensor(mask_dict[ith_e]).unsqueeze(0).to(device)
                e2_mask = torch.LongTensor(mask_dict[jth_e]).unsqueeze(0).to(device)
                e1_h = self.entity_average(sequence_output[idx,:].unsqueeze(0), e1_mask) # (batch_size,hidden_dim)
                e2_h = self.entity_average(sequence_output[idx,:].unsqueeze(0), e2_mask)
                # 每个实例出现的实体数量不同，得到的组合数量也不同，
                # 因此按照batch中每个实例数量不同扩展batch size,
                # 同时tags也要填充padding # 注意：tag id中需要事先加入无关系类别
                e1_h = self.activation(self.dense(e1_h))
                e2_h = self.activation(self.dense(e2_h))
                # [cls] + 实体1 + 实体2
                e1_typ_h = self.etype_emb(torch.tensor([self.etyp2id[entities[ith_e][0][0]]]).to(device))
                e2_typ_h = self.etype_emb(torch.tensor([self.etyp2id[entities[jth_e][0][0]]]).to(device))
                
                concat_h = torch.cat([pooled_output[idx,:].unsqueeze(0), e1_h, e1_typ_h, e2_h, e2_typ_h], dim=-1)
                concat_h = self.norm(concat_h)
                logits = self.classifier(self.drop(concat_h))
 
                # rlist: [(((obj_span,obj_typ),(subj_span,sub_typ),r)),...]
                e1_span_set = set([i[1:3] for i in entities[ith_e]])
                e2_span_set = set([i[1:3] for i in entities[jth_e]])
                  
                if cnt_cyc < 1:
                    logits_expanded = logits
                    logits_expanded_e_dict[idx].append((e1_span_set, e2_span_set, torch.argmax(logits, dim=1).item()))
                else:
                    logits_expanded = torch.cat([logits_expanded, logits], dim=0) # batch dimension
                    logits_expanded_e_dict[idx].append((e1_span_set, e2_span_set, torch.argmax(logits, dim=1).item()))
                
                for ir in rlists[idx]:
                    # 计算loss这里只考虑了被模型抽取出的实体形成的关系的预测
                    # 没有考虑因为实体抽取不全，而导致的关系不准确。
                    if (e1_span_set,e2_span_set) == (set(ir[0][0]),set(ir[1][0])) or (e1_span_set,e2_span_set) == (set(ir[1][0]),set(ir[0][0])):
                        rtag_ids.append([rtag2id[ir[2]]])
                    elif e1_span_set.intersection(set(ir[0][0])) and e1_span_set != set(ir[0][0]):
                        print("2*"*10)
                        print(e1_span_set)
                        print(entities[ith_e])
                        print(set(ir[0][0]))
                        print(ir)
                        tmp = token_ids[idx].tolist()[1:-1]
                        print([tmp[i:j] for i,j in set(ir[0][0])])
#                         sys.exit()
                    else:
                        rtag_ids.append([rtag2id['norelation']])
                cnt_cyc += 1
        nerloss = self.bert_ner.neg_log_likelihood(token_ids, token_type_ids, att_mask, pos_ids, nertag_ids, crf_mask)
#       print("logits_expanded.shape: ",logits_expanded.shape)
#       print("torch.LongTensor(rtag_ids).shape: ",torch.LongTensor(rtag_ids).shape)
        try:
            rloss = self.criterion(logits_expanded, torch.LongTensor(rtag_ids).squeeze(1).to(device))   
            rloss = rloss/logits_expanded.shape[0]
        except:
            rloss = torch.tensor(10000.0)
#             print("="*10)
#             print("self.loss_weight: ",self.loss_weight)
#             return nerloss/bs + self.loss_weight * rloss, logits_expanded_e_dict
        return nerloss/bs + rloss, logits_expanded_e_dict

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

def cal_metrics_from_tuples(pred_dict, true_tuples_lst):
    ner_acc_mean, ner_rec_mean = 0,0
    re_acc_mean, re_rec_mean = 0,0
    for i_instance, ituple_lst in enumerate(true_tuples_lst): # loop over one batch; i is for ith instance
        pred_e_lst = list()
        for k in pred_dict[i_instance]:  # 此instance中所有预测的实体
            pred_e_lst.extend([ii for ii in k[0]])
            pred_e_lst.extend([ii for ii in k[1]])
        true_e_lst = list() # 此instance中所有true实体
        for j in ituple_lst:
            true_e_lst.extend(j[0][0])
            true_e_lst.extend(j[1][0])
        # 计算实体抽取metrics
        ner_acc_mean += len(set(pred_e_lst).intersection(set(true_e_lst)))/(len(set(pred_e_lst)) + 0.00001) # 防止除0    
        ner_rec_mean += len(set(pred_e_lst).intersection(set(true_e_lst)))/(len(set(true_e_lst)) + 0.00001)
        
        # 计算（实体，实体，关系）全匹配metrics
        re_acc, re_rec = list(), list()
#         print(ituple_lst)
        ituple_lst = [(set(i[0][0]),set(i[1][0]),rtag2id[i[2]]) for i in ituple_lst]
#         print(ituple_lst)
#         sys.exit()
#         if len(pred_dict[i_instance])>0:
#             print("-"*20)
#             print("true")
#             print(ituple_lst)
#             print("pred")
#             print(pred_dict[i_instance])
        for pred_tuple in pred_dict[i_instance]:
            pred_e1_set, pred_e2_set, pred_r = pred_tuple[0],pred_tuple[1],pred_tuple[2]
#             print("1-"*10)
#             print(ituple_lst)
#             print(pred_e1_set,pred_e2_set,pred_r)
            if (pred_e1_set,pred_e2_set,pred_r) in ituple_lst or (pred_e2_set,pred_e1_set,pred_r) in ituple_lst:
                re_acc.append(1)
            else:
                re_acc.append(0)
        for true_tuple in ituple_lst:
#             true_e1_set, true_e2_set, true_r = set(true_tuple[0][0]),set(true_tuple[1][0]),rtag2id(true_tuple[2])
            true_e1_set, true_e2_set, true_r = true_tuple[0], true_tuple[1], true_tuple[2]
#             print("2-"*10)
#             print(true_e1_set, true_e2_set, true_r)
#             print(pred_dict[i_instance])
#             sys.exit()
            if (true_e1_set, true_e2_set, true_r) in pred_dict[i_instance] or (true_e1_set, true_e2_set, true_r) in pred_dict[i_instance]:
                re_rec.append(1)
            else:
                re_rec.append(0)
        re_acc_mean += sum(re_acc)/(len(re_acc)+0.00001)
        re_rec_mean += sum(re_rec)/(len(re_rec)+0.00001)
    bs = len(true_tuples_lst)
    ner_acc_mean = ner_acc_mean/bs
    ner_rec_mean = ner_rec_mean/bs
    ner_f1_mean = (2*ner_acc_mean*ner_rec_mean)/(ner_acc_mean+ner_rec_mean+0.00001)
    re_acc_mean = re_acc_mean/bs
    re_rec_mean = re_rec_mean/bs
    re_f1_mean = (2*re_acc_mean*re_rec_mean)/(re_acc_mean+re_rec_mean+0.00001)
    return ner_acc_mean,ner_rec_mean,ner_f1_mean,re_acc_mean,re_rec_mean,re_f1_mean
def get_group_parameters(model):
    params = list(model.named_parameters())
#     for n,p in params:
#         print(n)
# #     no_decay = ['bias,','LayerNorm']
# #     other = ['lstm','linear_layer']
# #     no_main = no_decay + other
# #     sys.exit()
#     print("params")
#     print(params)
    g1,g2,g3,g4,g5 = list(),list(),list(),list(),list()
    for n,p in params:
        if n.split(".")[0] in ["bert_ner"]:
            if n.split(".")[1] in ["bert"]:
                g1.append(p)
            elif n.split(".")[1] in ["crf"]:
                g2.append(p)
            else:
                g3.append(p)
        elif n.split(".")[0] in ["loss_weight"]:
            g4.append(p)
        else:
            g5.append(p)
    d1 = {'params':g1,'weight_decay':1e-3,'lr':1e-4} # bert encoding
    d2 = {'params':g2,'weight_decay':5e-3,'lr':1e-3} # crf
    d3 = {'params':g3,'weight_decay':5e-3,'lr':1e-3} # linear cls between bert encoding and crf
    d4 = {'params':g4,'weight_decay':1e-1,'lr':1e-2} # loss weight parameter
    d5 = {'params':g5,'weight_decay':2e-3,'lr':1e-3} # re cls
    print("Optimizer parameters: ")
    print("bert, weight_decay {},lr {}".format(d1["weight_decay"],d1["lr"]))
    print("hidden2tag, posemb, bertlinear weight_decay {},lr {}".format(d2["weight_decay"],d2["lr"]))
    print("crf, weight_decay {},lr {}".format(d3["weight_decay"],d3["lr"]))
    return [d1,d2,d3,d4]

if __name__ == "__main__":
  global nertag2id
  global pos2id
  global rtag2id
  global id2nertag
  global device

  bert_tokenizer_path = "/disc1/juan.du/bert_models/albert_chinese_tiny" 
#   bert_tokenizer_path = "/home/juan.du/nlp/bert_models/bert-base-chinese" 
  bert_model_path = "/disc1/juan.du/bert_models/albert_chinese_tiny"
#   bert_model_path = "/home/juan.du/nlp/bert_models/bert-base-chinese"
  bert_config = AutoConfig.from_pretrained(bert_model_path) 

  print("Args: {}".format(args))
  use_cuda = torch.cuda.is_available() and not args.no_cuda
  device = torch.device('cuda:0' if use_cuda else 'cpu')
#   device='cpu'
  torch.manual_seed(args.seed)
  if use_cuda:
    torch.cuda.manual_seed(args.seed)
  
  train_texts, train_nertag_lists, train_postags, train_rlists, pos2id, nertag2id, rtag2id, etyp2id = build_corpus(data_file=args.trainset, make_vocab=True)
  dev_texts, dev_nertag_lists, dev_postags, dev_rlists = build_corpus(data_file=args.devset, make_vocab=False)
  id2nertag = {v:k for k,v in nertag2id.items()}
#   print("nertag2id")
#   print(nertag2id)
  print("rtag2id")
  print(rtag2id)
#   print("pos2id")
#   print(pos2id)
#   print("train_texts")
#   print(train_texts[:2])
#   print("train_nertag_lists")
#   print(train_nertag_lists[:2])
#   print("train_postags")
#   print(train_postags[:2])
#   print("train_rlists")
#   print(len(train_rlists))
#   print(train_rlists[:30])
#   sys.exit()

#   for idx,_ in enumerate(range(2)):
#     for i,j in enumerate(train_texts[idx]):
#       print(j,train_tags[idx][i])
#     print("-"*10)
#     for i,j in enumerate(train_texts[idx]):
#       print(j,train_postags[idx][i])
#     print("="*10)
#   sys.exit()

  trainset = REDataset(train_texts, train_postags, train_nertag_lists, train_rlists, pos2id, nertag2id, rtag2id, bert_tokenizer_path)
  devset = REDataset(dev_texts, dev_postags, dev_nertag_lists, dev_rlists, pos2id, nertag2id, rtag2id, bert_tokenizer_path)
  print("train_rlists: ",train_rlists[:2])
  trainset_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=False)
  devset_loader = DataLoader(devset, batch_size=args.dev_batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=False)

  print("Building model")
  model = SentenceRE(bert_model_path, bert_config, len(nertag2id), len(pos2id), etyp2id).to(device)
  print(model)
  
  grouped_paras = get_group_parameters(model)

  eps = 1e-5
  warmup_steps = 100
  optimizer = AdamW(grouped_paras,eps=eps)
  t_total = len(trainset)//args.train_batch_size * args.epochs
  
  print("eps: ", eps)
  print("warmup_steps: ",warmup_steps)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

  print("Start training")
  model.train()
  step = 0
  
  best_f1 = 0
  best_prec = 0
  best_rec = 0
  patience = 0
  early_stop = False
  for eidx in range(1, args.epochs + 1):
    if eidx == 2:
      model.debug = True
    if early_stop:
      print("Early stop. epoch {} step {} best f1 {} precision {} recall {}".format(eidx, step, best_f1, best_prec, best_rec))
      sys.exit(0)
    print("Start epoch {}".format(eidx))
    for bidx, batch in enumerate(trainset_loader):
      optimizer.zero_grad()
      token_ids, token_type_ids, att_mask, pos_ids, nertags, crf_mask, rtag_ids = batch
      
      token_ids = token_ids.to(device)
      token_type_ids = token_type_ids.to(device)
      att_mask = att_mask.to(device)
      pos_ids = pos_ids.to(device)
      nertags = nertags.to(device)
      crf_mask = crf_mask.to(device)
#       rtag_ids = rtag_ids.to(device)

      loss, pred_dict = model.cal_loss(token_ids, token_type_ids, att_mask, pos_ids, crf_mask, nertags, rlists=rtag_ids)
      train_ner_acc, train_ner_rec, train_ner_f1, train_re_acc, train_re_rec, train_re_f1 = cal_metrics_from_tuples(pred_dict, rtag_ids)

      loss.backward()
      train_loss = loss.item()/args.train_batch_size
      optimizer.step()
      scheduler.step()
      step += 1
      if step % args.log_interval == 0:
        print("epoch {} step {} batch {} train loss {} ner f1 {} precision {} recall {}".format(eidx, step, bidx, round(train_loss,6),round(train_ner_f1,6),round(train_ner_acc,6),round(train_ner_rec,6)))
        print("re f1 {} precision {} recall {}".format(round(train_re_f1,6),round(train_re_acc,6),round(train_re_rec,6)))
      if step % args.save_interval == 0:
        torch.save(model.state_dict(), os.path.join(args.model_path, "newest.model"))
        torch.save(optimizer.state_dict(), os.path.join(args.model_path, "newest.optimizer"))
      if step % args.valid_interval == 0:
        model.eval()
        eval_loss_lst = []
        with torch.no_grad():
          eval_ner_acc_lst, eval_ner_rec_lst, eval_ner_f1_lst, eval_re_acc_lst, eval_re_rec_lst, eval_re_f1_lst = [],[],[],[],[],[]
          for bidx, batch in enumerate(devset_loader):
            token_ids, token_type_ids, att_mask, pos_ids, nertags, crf_mask, rtag_ids = batch
            # 去掉tags pad
            token_ids = token_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            att_mask = att_mask.to(device)
            pos_ids = pos_ids.to(device)
            crf_mask = crf_mask.to(device)
            
            loss, pred_dict = model.cal_loss(token_ids, token_type_ids, att_mask, pos_ids, crf_mask, nertag_ids=nertags, rlists=rtag_ids)
            eval_ner_acc, eval_ner_rec, eval_ner_f1, eval_re_acc, eval_re_rec, eval_re_f1 = cal_metrics_from_tuples(pred_dict, rtag_ids)
            eval_ner_acc_lst.append(eval_ner_acc)
            eval_ner_rec_lst.append(eval_ner_rec)
            eval_ner_f1_lst.append(eval_ner_f1)
            eval_re_acc_lst.append(eval_re_acc)
            eval_re_rec_lst.append(eval_re_rec)
            eval_re_f1_lst.append(eval_re_f1)
            eval_loss_lst.append(loss.item()/args.dev_batch_size)
        
        eval_re_f1_mean = sum(eval_re_f1_lst)/len(eval_re_f1_lst)
        eval_re_acc_mean = sum(eval_re_acc_lst)/len(eval_re_acc_lst)
        eval_re_rec_mean = sum(eval_re_rec_lst)/len(eval_re_rec_lst)
        print("[valid] epoch {} step {} dev mean loss {} ner f1 {} precision {} recall {}".format(eidx, step, round(sum(eval_loss_lst)/len(eval_loss_lst),6),\
        round(sum(eval_ner_f1_lst)/len(eval_ner_f1_lst),6),\
        round(sum(eval_ner_acc_lst)/len(eval_ner_acc_lst),6),\
        round(sum(eval_ner_rec_lst)/len(eval_ner_rec_lst),6)))
        print("re f1 {} precision {} recall {}".format(round(eval_re_f1_mean,6), round(eval_re_acc_mean,6), round(eval_re_rec_mean,6)))
        
        if eval_re_f1_mean > best_f1:
          patience = 0
          best_f1 = eval_re_f1_mean
          best_prec = eval_re_acc_mean
          best_rec = eval_re_rec_mean
          torch.save(model.state_dict(), os.path.join(args.model_path, "best.model"))
          torch.save(optimizer.state_dict(), os.path.join(args.model_path, "best.optimizer"))
        else:
          patience += 1
          if patience == args.patience:
            early_stop = True
  
  
  