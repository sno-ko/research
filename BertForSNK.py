import torch
from torch import nn
import numpy as np
import collections
from collections import defaultdict
import sys
import torch.nn.functional as F
import itertools
import math
import numpy as np
sys.path.append("../")
sys.path.append("../LSTM")
import LSTM

def myzip(*lst):
    s = len(lst[0])
    if any(len(v) != s for v in lst):
        raise Exception("イテレータのサイズが不一致 ({})".format([len(v) for v in lst]))
    return zip(*lst)

class BertForSNK(nn.Module):
    # オリジナルのにmode_docVecとlabelStr_dictを追加
    def __init__(self, net_bert, config_bert, labelStr_dict, fields_dict, mode_docVec, mode_stcVec,
                 flag_attn_mask=False, flag_multitask=False,
                 flag_two_level=False, flag_share=True,
                 mode_target_use=None, flag_multi_TA=False, mode_BP_ML="join", input_ids_target=None):
        super(BertForSNK, self).__init__()
        
        mode_target_use = mode_target_use if mode_target_use else ""

        # BERTモジュール
        self.bert = net_bert  # BERTモデル
        self.labelStr_dict = labelStr_dict
        self.labelStr_list = list(labelStr_dict.values())[0] if not flag_multitask else None
        self.flag_multitask = flag_multitask
        self.fields_dict = fields_dict
        self.mode_docVec = mode_docVec
        self.mode_stcVec = mode_stcVec
        self.flag_attn_mask = flag_attn_mask
        self.flag_two_level = flag_two_level
        self.flag_share = flag_share
        
        self.mode_target_use = mode_target_use
        self.flag_multi_TA = flag_multi_TA
        self.mode_BP_ML = mode_BP_ML
        self.num_target_attn_layer = None
        self.repeat_target_emb = None
        self.target_attn_layer = None
        self.target_encoder = None
        self.input_ids_target = None
        self.lstm = None
        
        if mode_docVec == "auto":
            raise Exception("非対応")
        
        # 課題文アテンション
        if mode_target_use:
            self.input_ids_target = input_ids_target
            self.repeat_target_emb = config_bert.repeat_target_emb
            
            if "LSTM" in mode_target_use:
                encoder_LSTM = LSTM.LSTM(config_bert.vocab_size, config_bert.hidden_size, 
                                    config_bert.hidden_size, None, 
                                    two_lstm=False, bidirectional=True, 
                                    dropout_rate=config_bert.hidden_dropout_prob, 
                                    mode_attn=None, attn_mask=True, use_liner=False)                
                
            if "target_attn-out" in mode_target_use:
                if "LSTM" in mode_target_use:
                    self.target_attn_layer = multiTargetAttantion(flag_multi_TA, labelStr_dict, encoder_LSTM, config_bert, input_ids_target)
                else:
                    self.target_attn_layer = multiTargetAttantion(flag_multi_TA, labelStr_dict, self.bert, config_bert, input_ids_target)
            elif "target_attn-in" in mode_target_use:
                if "LSTM" in mode_target_use:
                    self.lstm = encoder_LSTM
                self.num_target_attn_layer = config_bert.num_target_attn_layer
            elif "target_encode" in mode_target_use:
                self.target_encoder = TargetEncoder(self.bert, input_ids_target, mode_target_use)
            else:
                raise Exception("unknown mode_taget_use ({})".format(mode_target_use))
            
        hidden_dim = config_bert.hidden_size
        dropout_rate = config_bert.hidden_dropout_prob
        
        # 文分類層
        if flag_two_level:
            if not flag_share:
                if mode_target_use == "target_encode-cat":
                    self.clss_sub = multiHead(hidden_dim * 2, labelStr_dict, flag_multi_TA)
                else:
                    self.clss_sub = multiHead(hidden_dim, labelStr_dict, flag_multi_TA)
#                 nn.init.normal_(self.cls_sub.weight, std=0.02)
#                 nn.init.normal_(self.cls_sub.bias, 0)
            if "w_stc" in mode_docVec:
                if mode_target_use == "target_encode-cat":
                    self.stc_weight_layers = multiWeightedStc(labelStr_dict, mode_docVec, hidden_dim * 2, dropout_rate)
#                     self.stc_weight_layer = WeightedStc(mode_docVec, hidden_dim * 2, dropout_rate)
                else:
                    self.stc_weight_layers = multiWeightedStc(labelStr_dict, mode_docVec, hidden_dim, dropout_rate)
#                     self.stc_weight_layer = WeightedStc(mode_docVec, hidden_dim, dropout_rate)
        
        # 文書分類層
        if flag_two_level and not(flag_share) and mode_docVec == "auto":
            pass
        else:
            if mode_target_use == "target_encode-cat":
                self.clss = multiHead(hidden_dim * 2, labelStr_dict, flag_multi_TA)
#                 self.cls = nn.Linear(in_features=hidden_dim * 2, out_features=len(self.labelStr_list))
            else:
                self.clss = multiHead(hidden_dim, labelStr_dict, flag_multi_TA)
#                 self.cls = nn.Linear(in_features=hidden_dim, out_features=len(self.labelStr_list))

            #重み初期化処理
            #nn.init.kaiming_normal_(self.cls.weight)
#             nn.init.normal_(self.cls.weight, std=0.02)
#             nn.init.normal_(self.cls.bias, 0)

    def forward(self, input_ids, lengths=None, topics=None, attrs=None, output_all_encoded_layers=False, attention_show_flg=False):
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        token_type_ids： [batch_size, sequence_length]の、各単語が1文目なのか、2文目なのかを示すid
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：最終出力に12段のTransformerの全部をリストで返すか、最後だけかを指定
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''
        
        token_type_ids = None
        
        if self.flag_attn_mask:
            attention_mask = (input_ids != 0).int()
        else:
            attention_mask = None

        target_embs = None
        attention_mask_target = None
        if "target_attn-in" in self.mode_target_use:
            target_embs, attention_mask_target = self.get_target_embs(topics, input_ids.device)
        
        #BERTの基本モデル部分の順伝搬
        #順伝搬させる
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            encoded_layers, pooled_output, attention_probs = self.bert(
                input_ids, topics=topics, target_tensor=target_embs, attention_mask_target=attention_mask_target,
                token_type_ids=token_type_ids, attention_mask=attention_mask,
                output_all_encoded_layers=output_all_encoded_layers, attention_show_flg=attention_show_flg)
        elif attention_show_flg == False:
            encoded_layers, pooled_output = self.bert(
                input_ids, topics=topics, target_tensor=target_embs, attention_mask_target=attention_mask_target,
                token_type_ids=token_type_ids, attention_mask=attention_mask,
                output_all_encoded_layers=output_all_encoded_layers, attention_show_flg=attention_show_flg)
            
        if "target_attn-out" in self.mode_target_use:
            encoded_layers, attn_probs_target = self.target_attn_layer(encoded_layers, topics, attrs, True)
        elif "target_attn-in" in self.mode_target_use:
            attn_probs_target = self.bert.encoder.layer[self.num_target_attn_layer - 1].attention.target_attn.attn_weight
        else:
            encoded_layers = [encoded_layers]
        
        # 文書ベクトル
        if self.mode_docVec == "max" or self.mode_stcVec == "max":
            vecs = [self.max_pooling(input_ids, encoded_layers_i) for encoded_layers_i in encoded_layers]
        elif self.mode_docVec == "min" or self.mode_stcVec == "min":
            vecs = [self.min_pooling(input_ids, encoded_layers_i) for encoded_layers_i in encoded_layers]
        elif self.mode_docVec == "ave" or self.mode_stcVec == "ave":
            vecs = [self.ave_pooling(input_ids, encoded_layers_i) for encoded_layers_i in encoded_layers]
        elif self.mode_docVec == "CLS" or self.mode_stcVec == "CLS":
            vecs = [self.cls_extract(encoded_layers_i) for encoded_layers_i in encoded_layers]
        elif self.mode_docVec == "ave_stc":
            vecs = [self.ave_stc(input_ids, encoded_layers_i) for encoded_layers_i in encoded_layers]
        elif self.mode_docVec == "max_stc":
            vecs = [self.max_stc(input_ids, encoded_layers_i) for encoded_layers_i in encoded_layers]
        elif "w_stc" in self.mode_docVec:
            raise Exception("非対応")
        else:
            raise Exception("error: unknown method ({})".format(self.mode_docVec))

#             vecs = [v.view(-1, 768) for v in vecs]  #sizeを[batch_size, hidden_sizeに変換

        if "target_encode" in self.mode_target_use:
            vecs = [self.target_encoder(vecs_i, topics) for vecs_i in vecs]

        vecs = torch.stack(vecs, dim=0)

        if self.flag_two_level:
            if self.flag_share:
                outs_sub = self.clss(vecs, attrs)
            else:
                outs_sub = self.clss_sub(vecs, attrs)

            vecs_doc = []
            total_length = 0
            weight_stc_list = []
            preds_sub = []
            for length in lengths:
                vecs_stc = vecs[:, total_length: total_length + length]
                
                if self.mode_docVec == "ave_stc":
                    vec = vecs_stc.mean(dim=1)
                elif self.mode_docVec == "max_stc":
                    vec = vecs_stc.max(dim=1)[0]
                elif "w_stc" in self.mode_docVec:
                    vec, weights = self.stc_weight_layers(vecs_stc, attrs, return_attn=True)
                    weight_stc_list.append(weights)
                elif self.mode_docVec == "auto":
                    vec = None
                    pred = out_sub[:, total_length: total_length + length].max(2)[1]
                    preds_sub.append(pred)
                
                vecs_doc.append(vec)
                total_length += length

            if self.mode_docVec == "auto":
                vecs_doc = [[None] * len(preds_sub[0]) for _ in range(len(preds_sub))]
                out = self.pred_doc_auto(preds_sub)
            else:
                vecs_doc = torch.stack(vecs_doc, dim=1)
                outs = self.clss(vecs_doc, attrs)
                
            result = [outs, vecs_doc, outs_sub, vecs]
        else:
            outs = self.clss(vecs, attrs)
            result = [outs, vecs]

        #attention_showのときは、attention_probs（1番最後の）もリターンする
        if attention_show_flg:
            result.append(attention_probs)
            
            if "w_stc" in self.mode_docVec:
                result.append(weight_stc_list)
                
            if "target_attn" in self.mode_target_use:
                result.append(attn_probs_target)
        
        return result
    
    def pred_doc_auto(self, preds_sub):
        if self.multitask:
            raise Exception("この関数はマルチタスクラーニングに非対応")
        one_hot_vec_list = np.diag([1.0] * len(self.labelStr_list)).tolist()
        def judge(l):
            max_label = l.max()
            if max_label <= 1:
                doc_label = max_label
            else:
                if max_label == 2 and 1 in l:
                    doc_label = 3
                else:
                    doc_label = max_label
            return one_hot_vec_list[int(doc_label)]
        preds_doc = torch.tensor([judge(preds) for preds in preds_sub],
                                 dtype=torch.float32, device=preds_sub[0].device)
        return preds_doc

    def get_target_embs(self, topics, device):
        if "LSTM" in self.mode_target_use:
            if self.repeat_target_emb:
                raise Exception("未実装")
            else:
                # 複製して埋め込み
                inputs_t = torch.stack([self.input_ids_target[t] for t in topics], dim=0).to(device=device)
                target_embs = self.lstm(inputs_t, reduce=False)
                attention_mask = (inputs_t != 0).int()
                return target_embs, attention_mask
        else:
            if self.repeat_target_emb:
                # 埋め込んで複製
                inputs_t = self.input_ids_target.to(device=device)
                attention_mask = (inputs_t != 0).int()
                target_embs, _ = self.bert(inputs_t, output_all_encoded_layers=False, attention_mask=attention_mask)
                target_embs_ex = torch.stack([target_embs[t] for t in topics], dim=0).to(device=device)
    #             attention_mask = (inputs_t > 4).int()
                attention_mask_ex = torch.stack([attention_mask[t] for t in topics], dim=0).to(device=device)
                return target_embs_ex, attention_mask_ex
            else:
                # 複製して埋め込み
                inputs_t = torch.stack([self.input_ids_target[t] for t in topics], dim=0).to(device=device)
                attention_mask = (inputs_t != 0).int()
                target_embs, _ = self.bert(inputs_t, output_all_encoded_layers=False, attention_mask=attention_mask)
    #             attention_mask = (inputs_t > 4).int()
                return target_embs, attention_mask
    
    def max_pooling(self, input_ids, encoded_layers):
        out = []
        for input_ids_i, encoded_layers_i in myzip(input_ids, encoded_layers):
            vec = encoded_layers_i[input_ids_i > 0].max(dim=0)[0]
            out.append(vec)
        out = torch.stack(out, dim=0).to(device=encoded_layers.device)
        return out

    def min_pooling(self, input_ids, encoded_layers):
        out = []
        for input_ids_i, encoded_layers_i in myzip(input_ids, encoded_layers):
            vec = encoded_layers_i[input_ids_i > 0].min(dim=0)[0]
            out.append(vec)
        out = torch.stack(out, dim=0).to(device=encoded_layers.device)
        return out
    
    def ave_pooling(self, input_ids, encoded_layers):
        out = []
        for input_ids_i, encoded_layers_i in myzip(input_ids, encoded_layers):
            vec = encoded_layers_i[input_ids_i > 0].mean(dim=0)
            out.append(vec)
        out = torch.stack(out, dim=0).to(device=encoded_layers.device)
        return out
    
    def cls_extract(self, encoded_layers):
        return encoded_layers[:, 0, :]

#     def ave_stc(self, input_ids, encoded_layers):
#         vecs = torch.stack([doc[i == self.ID_for_ssc].mean(dim=0) for i, doc in myzip(input_ids, encoded_layers)], dim=0)
#         return vecs

#     def max_stc(self, input_ids, encoded_layers):
#         vecs = torch.stack([doc[i == self.ID_for_ssc].max(dim=0)[0] for i, doc in myzip(input_ids, encoded_layers)], dim=0)
#         return vecs
    
# マルチタスクラーニングに対応したクラス分類層
class multiHead(nn.Module):
    def __init__(self, hidden_dim, labelStr_dict, flag_multi_input_vec=False):
        super(multiHead, self).__init__()
        self.nets = [] # 分類器のリスト
        self.attrs = list(labelStr_dict.keys())
        self.flag_multi_input_vec = flag_multi_input_vec
        for index_attr, (attr, labels) in enumerate(labelStr_dict.items(), 1):
            net = nn.Linear(in_features=hidden_dim, out_features=len(labels))
            nn.init.normal_(net.weight, std=0.02)
            nn.init.normal_(net.bias, 0)
            self.nets.append(net)
            # 属性としても付与
            setattr(self, "net_{}".format(index_attr), net)
        
    def forward(self, vec, attrs):
        # 各タスクの予測をリストに
        if self.flag_multi_input_vec:
            if len(attrs) != len(vec):
                raise Exception("入力行列数と分類器の数が違う")
            outputs = [self.nets[index_attr](vec_i) for index_attr, vec_i in myzip(attrs, vec)]
        else:
            outputs = [self.nets[index_attr](vec[0]) for index_attr in attrs]
        return outputs
    
class WeightedStc(nn.Module):
    def __init__(self, mode_docVec, hidden_dim, dropout_rate=None):
        super(WeightedStc, self).__init__()
        
        self.mode_docVec = mode_docVec
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        if mode_docVec == "w_stc":
            self.main = nn.Linear(hidden_dim, 1)
        elif mode_docVec == "w_stc-2":
            dim = int(hidden_dim / 2)
            self.main = nn.Sequential(
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout_rate),
                nn.Tanh(),
                nn.Linear(dim, 1)
            ) 
        
    def get_weight(self, vecs_stc):
        weights = self.main(vecs_stc)
        weights = F.softmax(weights, dim=-2)
        return weights

    def forward(self, vecs_stc, return_attn=False):
        weights = self.get_weight(vecs_stc)
        vec = (weights * vecs_stc).sum(dim=-2)
        if return_attn:
            return vec, weights
        else:
            return vec
    
class multiWeightedStc(nn.Module):
    def __init__(self, labelStr_dict, mode_docVec, hidden_dim, dropout_rate=None):
        super(multiWeightedStc, self).__init__()
        self.nets = []
        self.attrs = list(labelStr_dict.keys())
        
        for index_attr in range(1, len(self.attrs) + 1):
            net = WeightedStc(mode_docVec, hidden_dim, dropout_rate)
            self.nets.append(net) 
            # 属性としても付与
            setattr(self, "net_{}".format(index_attr), net)
            
    def forward(self, vecs_stc, attrs, return_attn=False):
        if return_attn:
            vec, weight = list(myzip(*[self.nets[index_attr](vecs_stc[i] if len(vecs_stc) > 1 else vecs_stc[0], return_attn) for i, index_attr in enumerate(attrs)]))
            out = (torch.stack(vec), torch.stack(weight))
        else:
            out = [self.nets[index_attr](vecs_stc[i] if len(vecs_stc) > 1 else vecs_stc[0], return_attn) for i, index_attr in enumerate(attrs)]
        return out
        
class TargetAttention(nn.Module):
    def __init__(self, encoder, config, input_ids_target):
        super(TargetAttention, self).__init__()
        
        self.mode_target_use = config.mode_target_use
        self.repeat_target_emb = config.repeat_target_emb
        self.flag_drop = "drop" in self.mode_target_use
        self.flag_norm = "norm" in self.mode_target_use
        
        self.encoder = encoder
        self.input_ids_target = input_ids_target
        self.attention_mask = (input_ids_target != 0).int()
        self.attention_mask_inf = (1.0 - self.attention_mask) * -10000.0
        
        hidden_dim = config.hidden_size
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        if self.flag_drop:
            self.dropout_1 = nn.Dropout(config.hidden_dropout_prob)
            
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        
        if self.flag_drop:
            self.dropout_2 = nn.Dropout(config.hidden_dropout_prob)
            
        if self.flag_norm:
            self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs, topics, return_attn=False):
        if "LSTM" in self.mode_target_use:
            if self.repeat_target_emb:
                # 埋め込んで複製
                raise Exception("未実装")
            else:
                # 複製して埋め込み
                input_ids_target = torch.stack([self.input_ids_target[t] for t in topics], dim=0).to(device=inputs.device)
                target_vecs = self.encoder(input_ids_target, reduce=False)
                attn_mask = (input_ids_target != 0).int()
                attn_mask = ((1.0 - attn_mask.to(dtype=torch.float32)) * -10000.0).unsqueeze(1)
        else:
            if self.repeat_target_emb:
                # 埋め込んで複製
                encoded_layers_target, _ = self.encoder(
                    self.input_ids_target.to(device=inputs.device), output_all_encoded_layers=False,
                    attention_mask=self.attention_mask.to(device=inputs.device))

                target_vecs = torch.stack([encoded_layers_target[t] for t in topics], dim=0)
                attn_mask = torch.stack([self.attention_mask_inf[t] for t in topics], dim=0).unsqueeze(1)
                attn_mask = attn_mask.to(device=inputs.device, dtype=inputs.dtype)
            else:
                # 複製して埋め込み
                input_ids_target = torch.stack([self.input_ids_target[t] for t in topics], dim=0).to(device=inputs.device)
                attn_mask = (input_ids_target != 0).int()
                target_vecs, _ = self.encoder(
                    input_ids_target, output_all_encoded_layers=False, attention_mask=attn_mask)
                attn_mask = ((1.0 - attn_mask.to(dtype=torch.float32)) * -10000.0).unsqueeze(1)
        
        if "CLS" in self.mode_target_use:
            inputs_q = self.query(inputs[:, :1])
        else:
            inputs_q = self.query(inputs)
            
        target_vecs_k = self.key(target_vecs)
        target_vecs_v = self.value(target_vecs)
        
        attn_score = torch.matmul(inputs_q, target_vecs_k.transpose(-1, -2))
        attn_score = attn_score / math.sqrt(self.hidden_dim)
        attn_score = attn_score + attn_mask
        attn_weight = F.softmax(attn_score, dim=-1)
        
        if self.flag_drop:
            attn_weight = self.dropout_1(attn_weight)
        
        response = torch.matmul(attn_weight, target_vecs_v)
        response = self.dense(response)
        
        if self.flag_drop:
            response = self.dropout_2(response)
        
        if "CLS" in self.mode_target_use:
            inputs[:, :1, :] = response + inputs[:, :1, :]
            out = inputs
        else:
            out = response + inputs
            
        if self.flag_norm:
            out = self.norm(out)
        
        if return_attn:
            return out, attn_weight
        else:
            return out

class multiTargetAttantion(nn.Module):
    def __init__(self, flag_multi_TA, labelStr_dict, encoder, config, input_ids_target):
        super(multiTargetAttantion, self).__init__()
        self.nets = []
        self.flag_multi_TA = flag_multi_TA
        self.attrs = list(labelStr_dict.keys())
        self.num_TA = len(labelStr_dict) if flag_multi_TA else 1
        
        for index_attr in range(1, self.num_TA + 1):
            net = TargetAttention(encoder, config, input_ids_target)
            self.nets.append(net)
            # 属性としても付与
            setattr(self, "net_{}".format(index_attr), net)
            
    def forward(self, inputs, topics, attrs, return_attn=False):
        if return_attn:
            outputs, attn_weight = list(myzip(*[self.nets[index_attr if self.num_TA > 1 else 0](inputs, topics, return_attn) for index_attr in attrs]))
            return outputs, attn_weight
        else:
            outputs = [self.nets[index_attr if self.num_TA > 1 else 0](inputs, topics, return_attn) for index_attr in attrs]
            outputs = torch.stack(outputs, dim=0)
            return outputs
            
            
# multi-head版
# class TargetAttention(nn.Module):
#     def __init__(self, bert, config, input_ids_target):
#         super(TargetAttention, self).__init__()
        
#         self.bert = bert
#         self.input_ids_target = input_ids_target
#         self.attention_mask = (input_ids_target != 0).int()
#         self.attention_mask_inf = (1.0 - self.attention_mask) * -10000.0
                
#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 768/12=64
#         self.all_head_size = self.num_attention_heads * self.attention_head_size  # = 'hidden_size': 768
        
#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.dropout_1 = nn.Dropout(config.hidden_dropout_prob)
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dropout_2 = nn.Dropout(config.hidden_dropout_prob)
#         self.norm = nn.LayerNorm(config.hidden_size)
        
#     def transpose_for_scores(self, x):
#         '''multi-head Attention用にテンソルの形を変換する
#         [batch_size, seq_len, hidden] → [batch_size, 12, seq_len, hidden/12] 
#         '''
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)
    
#     def forward(self, inputs, topics, return_attn=False):
#         # ターゲット文をエンコード
#         encoded_layers_target, _ = self.bert(
#             self.input_ids_target.to(device=inputs.device), output_all_encoded_layers=False,
#             attention_mask=self.attention_mask.to(device=inputs.device))
        
#         # ターゲット文, Attentionマスクを複製(&multi-head用に変形)
#         target_vecs = torch.stack([encoded_layers_target[t] for t in topics], dim=0)
#         attn_mask = torch.stack([self.attention_mask_inf[t] for t in topics], dim=0).unsqueeze(1)
#         attn_mask = attn_mask.to(device=inputs.device, dtype=inputs.dtype)
#         attn_mask = attn_mask.unsqueeze(2)
        
#         # query, key, value作成(&変形)
#         inputs_q = self.transpose_for_scores(self.query(inputs))
#         target_vecs_k = self.transpose_for_scores(self.key(target_vecs))
#         target_vecs_v = self.transpose_for_scores(self.value(target_vecs))
        
#         # Attention weight計算
#         attn_score = torch.matmul(inputs_q, target_vecs_k.transpose(-1, -2))
#         attn_score = attn_score / math.sqrt(self.attention_head_size)
#         attn_score = attn_score + attn_mask
#         attn_weight = F.softmax(attn_score, dim=-1)
        
#         # weightとvalueを掛ける
#         response = torch.matmul(attn_weight, target_vecs_v)
        
#         # multi-head Attentionのテンソルの形をもとに戻す
#         response = response.permute(0, 2, 1, 3).contiguous()
#         new_response_shape = response.size()[:-2] + (self.all_head_size,)
#         response = response.view(*new_response_shape)
        
#         response = self.dense(response)
#         response = self.dropout_2(response)

#         out = response + inputs
# #         out = self.norm(out)
        
#         if return_attn:
#             return out, attn_weight
#         else:
#             return out
        

class TargetEncoder(nn.Module):
    def __init__(self, bert, input_ids_target, mode_target_use):
        super(TargetEncoder, self).__init__()
        
        self.bert = bert
        self.input_ids_target = input_ids_target
        self.mode_target_use = mode_target_use
        self.attention_mask = (input_ids_target != 0).int()
        
        hidden_dim = bert.encoder.layer[0].output.dense.out_features
#         self.dense = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, inputs, topics):
        encoded_layers_target, _ = self.bert(
            self.input_ids_target.to(device=inputs.device), None, self.attention_mask.to(device=inputs.device), False, False)
        target_vecs = encoded_layers_target[:, 0, :]
#         target_vecs = self.dense(target_vecs)
        target_vecs = torch.stack([target_vecs[t] for t in topics], dim=0)
        
        if "max" in self.mode_target_use:
            out = torch.stack([inputs, target_vecs], dim=1).max(dim=1)[0]
        elif "ave" in self.mode_target_use:
            out = torch.stack([inputs, target_vecs], dim=1).mean(dim=1)            
        elif "cat" in self.mode_target_use:
            out = torch.cat([inputs, target_vecs], dim=1)
            
        return out

class TargetSpecificAttention(nn.Module):
    def __init__(self, bert, input_ids_target):
        super(TargetSpecificAttention, self).__init__()
        
        self.bert = bert
        self.input_ids_target = input_ids_target
        self.attention_mask = (input_ids_target != 0).int()
        
        hidden_dim = net_bert.encoder.layer[0].output.dense.out_features
        self.attn = nn.Linear(hidden_dim * 2, 1)
#         nn.init.normal_(self.attn.weight, std=0.02)
#         nn.init.normal_(self.attn.bias, 0)
    
    def forward(self, inputs, topics):
        encoded_layers_target, _ = self.bert(
            self.input_ids_target.to(device=inputs.device), 
            topics=None, attention_mask=self.attention_mask.to(device=inputs.device), 
            output_all_encoded_layers=False, attention_show_flg=False)
        
        target_vecs = encoded_layers_target[:, 0, :]
        target_vecs = torch.stack([target_vecs[t] for t in topics], dim=0)
        target_vecs_expanded = torch.unsqueeze(target_vecs, 1).expand([-1, inputs.shape[1], -1])
        
        joint_vecs = torch.cat([inputs, target_vecs_expanded], dim=-1)
        attn_score = self.attn(joint_vecs)
        attn_score = F.softmax(attn_score, dim=1)
        weighted_vecs = attn_score * inputs 
        out = weighted_vecs.sum(dim=1)
        
        return out
    
##########################################################################
    
# ネストなリストを平坦化する
def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

#前処理と単語分割をまとめた関数を定義
#単語分割の関数を渡すので、tokenizer_bertではなく、tokenizer_bert.tokenizeを渡す点に注意
def tokenizer_with_preprocessing(text, tokenizer):
    ret = tokenizer(text)
    return ret

# ssc用
def tokenizer_for_ssc(text, tokenizer, str_sep):
    ret = []
    for stc in text.split(str_sep):
        ret.extend(tokenizer(stc))
        ret.append(str_sep)
    ret.pop()
    return ret

def tokenizer_split_stc(text, str_sep):
    tokens = text.strip().split(str_sep)
    return tokens

def preprocess(tokens, token_for_ssc, str_cls, str_sep, str_sep_org):
    if token_for_ssc == "[SEP]":
        new_tokens = [word if str_sep_org not in word else str_sep for word in tokens]
    elif token_for_ssc == "[CLS]":
        new_tokens = [word if str_sep_org not in word else [str_sep, str_cls] for word in tokens]
        new_tokens = list(flatten(new_tokens))
    return new_tokens


