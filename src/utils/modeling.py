import torch
import torch.nn as nn
from transformers import BertModel, BartConfig, BartForSequenceClassification, BertForSequenceClassification, ModernBertModel
from transformers.models.bart.modeling_bart import BartEncoder, BartPretrainedModel
from transformers import BartPreTrainedModel
from transformers import BartModel, RobertaModel, DistilBertModel, DebertaModel
import torch.nn.functional as F
from transformers import BartModel, BartConfig, AutoConfig
from transformers import AutoModel, AutoModelForMaskedLM


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_attn = nn.Linear(encoder_hidden_dim, attention_dim)
        self.decoder_attn = nn.Linear(decoder_hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        encoder_energy = self.encoder_attn(encoder_outputs)  # (batch_size, seq_len, attention_dim)
        decoder_energy = self.decoder_attn(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)
        energy = torch.tanh(encoder_energy + decoder_energy)  # (batch_size, seq_len, attention_dim)
        attention_weights = torch.softmax(self.v(energy).squeeze(-1), dim=1)  # (batch_size, seq_len)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, encoder_hidden_dim)
        context_vector = context_vector.squeeze(1)  # (batch_size, encoder_hidden_dim)
        return context_vector, attention_weights
    
## ModernBert

class modern_bert_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(modern_bert_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        # self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
        # self.mbert = ModernBertModel.from_pretrained("answerdotai/ModernBERT-large")
        self.mbert = ModernBertModel.from_pretrained("answerdotai/ModernBERT-base")

        # self.encoder = self.bart.get_encoder()
        self.mbert.pooler = None
        self.linear = nn.Linear(self.mbert.config.hidden_size*2, self.mbert.config.hidden_size) #1
        self.out = nn.Linear(self.mbert.config.hidden_size, num_labels)

        # # Dropout
        # self.dropout = nn.Dropout(dropout) if gen == 0 else nn.Dropout(dropoutrest)
        # self.relu = nn.ReLU()
        
        # self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
        # self.encoder = BartModel.from_pretrained("facebook/bart-large-mnli").get_encoder()
        # # self.encoder.pooler = None
        # self.encoder.pooler = BartModel.from_pretrained("facebook/bart-large-mnli").pooler

        
        # # Linear layers
        # self.linear = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        # self.out = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']

        last_hidden = self.mbert(input_ids=x_input_ids, attention_mask=x_atten_masks).last_hidden_state
        
        # last_hidden = self.encoder(input_ids=x_input_ids, attention_mask=x_atten_masks).last_hidden_state
        
        eos_token_ind = x_input_ids.eq(self.mbert.config.eos_token_id).nonzero() # tensor([[0,4],[0,5],[0,11],[1,2],[1,3],[1,6]...])
        
        

        assert len(eos_token_ind) == 2*len(kwargs['input_ids'])
        b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i%2==0]
        e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i+1)%2==0]
        x_atten_clone = x_atten_masks.clone().detach()

        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[:begin+1] = 0, 0 # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0 # <s> --> 0; 3rd </s> --> 0
        
        txt_l = x_atten_masks.sum(1).to('cuda')
        topic_l = x_atten_clone.sum(1).to('cuda')
        

        txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = x_atten_clone.type(torch.FloatTensor).to('cuda')
        txt_mean = torch.einsum('blh,bl->bh', last_hidden, txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden, topic_vec) / topic_l.unsqueeze(1)

        cat = torch.cat((txt_mean, topic_mean), dim=1) #1
        # cat = torch.cat((txt_mean, topic_mean, txt_mean - topic_mean, txt_mean * topic_mean), dim=1)
        
        
        # raise Exception
        
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out

# # BERT

# class MB(nn.Module):

#     def __init__(self, num_labels, gen, dropout, dropoutrest):
#         super(MB, self).__init__()
#         self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
#         self.relu = nn.ReLU()

#         self.mb = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")
#         self.classifier = nn.Linear(self.mb.config.hidden_size, num_labels)

#     def forward(self, **kwargs):
#         x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
#         last_hidden = self.mb(input_ids=x_input_ids, attention_mask=x_atten_masks)
#         print(last_hidden.config)

 
class bert_classifier(nn.Module):

    def __init__(self, num_labels, gen, dropout, dropoutrest):

        super(bert_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.pooler = None

        self.linear = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size) #2

        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        
        x_atten_masks[:,0] = 0 # [CLS] --> 0 
        idx = torch.arange(0, last_hidden[0].shape[1], 1).to('cuda')
        x_seg_ind = x_seg_ids * idx
        x_att_ind = (x_atten_masks-x_seg_ids) * idx
        indices_seg = torch.argmax(x_seg_ind, 1, keepdim=True)
        indices_att = torch.argmax(x_att_ind, 1, keepdim=True)
        for seg, seg_id, att, att_id in zip(x_seg_ids, indices_seg, x_atten_masks, indices_att):
            seg[seg_id] = 0  # 2nd [SEP] --> 0 
            att[att_id:] = 0  # 1st [SEP] --> 0 
        
        txt_l = x_atten_masks.sum(1).to('cuda')
        topic_l = x_seg_ids.sum(1).to('cuda')
        txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = x_seg_ids.type(torch.FloatTensor).to('cuda')
        txt_mean = torch.einsum('blh,bl->bh', last_hidden[0], txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden[0], topic_vec) / topic_l.unsqueeze(1)
        
        cat = torch.cat((txt_mean, topic_mean), dim=1)
        
        # cat = torch.cat((txt_mean, topic_mean, txt_mean - topic_mean, txt_mean * topic_mean), dim=1) #2
        
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out

from transformers import AutoModel

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class bertweet_classifier(nn.Module):
    def __init__(self, num_labels, gen, dropout, dropoutrest):
        super(bertweet_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout) if gen == 0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()

        # self.config = AutoConfig.from_pretrained("vinai/bertweet-base")
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        # self.bertweet = BertForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=3)
        self.bertweet.pooler = None  # BERTweet does not use a pooler
        # print(self.bertweet.config)
        
        self.linear = nn.Linear(self.bertweet.config.hidden_size * 2, self.bertweet.config.hidden_size)  #1
        self.out = nn.Linear(self.bertweet.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs["input_ids"], kwargs["attention_mask"]

        # Debugging: Print shapes before passing into BERTweet
        # print(f"x_input_ids.shape: {x_input_ids.shape}")  
        # print(f"x_atten_masks.shape: {x_atten_masks.shape}")
        # print(f"x_input_ids.device: {x_input_ids.device}")  
        # print(f"x_atten_masks.device: {x_atten_masks.device}")
        
        # max_len = x_input_ids.size(1)  # length of the sequence
        # print(f"Max sequence length: {max_len}")
        # print(x_input_ids[0])
        # print(x_atten_masks[0])
        # print(x_input_ids)
        # # print(x_input_ids.device, x_atten_masks.device)
        # # print(type(x_input_ids), type(x_atten_masks))
        # print(x_atten_masks)
        # Ensure tensors are moved to the same device
        # assert not torch.isnan(x_input_ids).any(), "NaN found in input_ids"
        # assert not torch.isnan(x_atten_masks).any(), "NaN found in attention_mask"
        device = x_input_ids.device
        # # print(device)
        # print(device)


        # raise Exception


        # Forward pass through BERTweet
        # last_hidden = self.bertweet(input_ids=x_input_ids, attention_mask=x_atten_masks, output_hidden_states=True).hidden_states[-1]
        # print("SSMB"*10)
        last_hidden = self.bertweet(x_input_ids, x_atten_masks).last_hidden_state
        # print("SB"*10)
        
        # Use correct eos_token_id for BERTweet
        eos_token_id = self.bertweet.config.eos_token_id
        eos_token_ind = x_input_ids.eq(eos_token_id).nonzero()

        # Debugging: Check tensor lengths
        # print(f"len(eos_token_ind): {len(eos_token_ind)}")
        # print(f"len(x_input_ids): {len(x_input_ids)}, 3 * len(x_input_ids): {3 * len(x_input_ids)}")

        # Fix assertion to prevent crashes
        assert len(eos_token_ind) == 3 * len(x_input_ids), "Mismatch in EOS token count!"

        b_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if i % 3 == 0]
        e_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if (i + 1) % 3 == 0]
        
        x_atten_clone = x_atten_masks.clone().detach()

        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[: begin + 2] = 0, 0  # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0  # <s> --> 0; 3rd </s> --> 0

        # Convert attention masks to float and ensure they are on the same device
        txt_l = x_atten_masks.sum(1).to(device)
        topic_l = x_atten_clone.sum(1).to(device)
        txt_vec = x_atten_masks.float().to(device)
        topic_vec = x_atten_clone.float().to(device)

        txt_mean = torch.einsum("blh,bl->bh", last_hidden, txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum("blh,bl->bh", last_hidden, topic_vec) / topic_l.unsqueeze(1)

        # Concatenate sentence representations
        cat = torch.cat((txt_mean, topic_mean), dim=1)

        # Apply dropout, activation, and final classification layers
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out


class roberta_classifier(nn.Module):
    def __init__(self, num_labels, gen, dropout, dropoutrest):
        super(roberta_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout) if gen == 0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()

        # self.config = AutoConfig.from_pretrained("vinai/bertweet-base")
        self.roberta = RobertaModel.from_pretrained("FacebookAI/roberta-base")
        # self.bertweet = BertForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=3)
        self.roberta.pooler = None  # BERTweet does not use a pooler
        # print(self.bertweet.config)
        
        self.linear = nn.Linear(self.roberta.config.hidden_size * 2, self.roberta.config.hidden_size)  #1
        self.out = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs["input_ids"], kwargs["attention_mask"]

        # Debugging: Print shapes before passing into BERTweet
        # print(f"x_input_ids.shape: {x_input_ids.shape}")

        # print(f"x_atten_masks.shape: {x_atten_masks.shape}")

        # print(f"x_input_ids.device: {x_input_ids.device}")  
        # print(f"x_atten_masks.device: {x_atten_masks.device}")
        
        # max_len = x_input_ids.size(1)  # length of the sequence
        # print(f"Max sequence length: {max_len}")
        # print(x_input_ids[0])
        # print(x_atten_masks[0])
        # print(x_input_ids)
        # # print(x_input_ids.device, x_atten_masks.device)
        # # print(type(x_input_ids), type(x_atten_masks))
        # print(x_atten_masks)
        # Ensure tensors are moved to the same device
        # assert not torch.isnan(x_input_ids).any(), "NaN found in input_ids"
        # assert not torch.isnan(x_atten_masks).any(), "NaN found in attention_mask"

        device = x_input_ids.device
        # # print(device)
        # print(device)


        # raise Exception

        last_hidden = self.roberta(x_input_ids, x_atten_masks).last_hidden_state
        # print("SB"*10)
        
        # Use correct eos_token_id for Roberta
        eos_token_id = self.roberta.config.eos_token_id
        eos_token_ind = x_input_ids.eq(eos_token_id).nonzero()

        # Debugging: Check tensor lengths
        # print(f"len(eos_token_ind): {len(eos_token_ind)}")
        # print(f"len(x_input_ids): {len(x_input_ids)}, 3 * len(x_input_ids): {3 * len(x_input_ids)}")

        # Fix assertion to prevent crashes
        assert len(eos_token_ind) == 3 * len(x_input_ids), "Mismatch in EOS token count!"

        b_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if i % 3 == 0]
        e_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if (i + 1) % 3 == 0]
        
        x_atten_clone = x_atten_masks.clone().detach()

        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[: begin + 2] = 0, 0  # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0  # <s> --> 0; 3rd </s> --> 0

        # Convert attention masks to float and ensure they are on the same device
        txt_l = x_atten_masks.sum(1).to(device)
        topic_l = x_atten_clone.sum(1).to(device)
        txt_vec = x_atten_masks.float().to(device)
        topic_vec = x_atten_clone.float().to(device)

        txt_mean = torch.einsum("blh,bl->bh", last_hidden, txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum("blh,bl->bh", last_hidden, topic_vec) / topic_l.unsqueeze(1)

        # Concatenate sentence representations
        cat = torch.cat((txt_mean, topic_mean), dim=1)

        # Apply dropout, activation, and final classification layers
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out

class deberta_classifier(nn.Module):
    def __init__(self, num_labels, gen, dropout, dropoutrest):
        super(deberta_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout) if gen == 0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()

        # self.config = AutoConfig.from_pretrained("vinai/bertweet-base")
        self.deberta = DebertaModel.from_pretrained("microsoft/deberta-base")
        # self.bertweet = BertForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=3)
        self.deberta.pooler = None  # BERTweet does not use a pooler
        # print(self.bertweet.config)
        
        self.linear = nn.Linear(self.deberta.config.hidden_size * 2, self.deberta.config.hidden_size)  #1
        self.out = nn.Linear(self.deberta.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs["input_ids"], kwargs["attention_mask"]

        
        device = x_input_ids.device
        # print(self.deberta.config)
        # print(self.deberta.config.eos_token_id)
        # raise Exception

        last_hidden = self.deberta(x_input_ids, x_atten_masks).last_hidden_state
        # print("SB"*10)
        
        # Use correct eos_token_id for Roberta
        eos_token_id = 2
        eos_token_ind = x_input_ids.eq(eos_token_id).nonzero()
        

        # Debugging: Check tensor lengths
        # print(f"len(eos_token_ind): {len(eos_token_ind)}")
        # print(f"len(x_input_ids): {len(x_input_ids)}, 3 * len(x_input_ids): {3 * len(x_input_ids)}")

        # Fix assertion to prevent crashes
        assert len(eos_token_ind) == 2 * len(x_input_ids), "Mismatch in EOS token count!"

        b_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if i % 2 == 0]
        e_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if (i + 1) % 2 == 0]
        
        x_atten_clone = x_atten_masks.clone().detach()

        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[: begin + 1] = 0, 0  # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0  # <s> --> 0; 3rd </s> --> 0

        # Convert attention masks to float and ensure they are on the same device
        txt_l = x_atten_masks.sum(1).to(device)
        topic_l = x_atten_clone.sum(1).to(device)
        txt_vec = x_atten_masks.float().to(device)
        topic_vec = x_atten_clone.float().to(device)

        txt_mean = torch.einsum("blh,bl->bh", last_hidden, txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum("blh,bl->bh", last_hidden, topic_vec) / topic_l.unsqueeze(1)

        # Concatenate sentence representations
        cat = torch.cat((txt_mean, topic_mean), dim=1)

        # Apply dropout, activation, and final classification layers
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out

class distilbert_classifier(nn.Module):
    def __init__(self, num_labels, gen, dropout, dropoutrest):
        super(distilbert_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout) if gen == 0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()

        # self.config = AutoConfig.from_pretrained("vinai/bertweet-base")
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # self.bertweet = BertForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=3)
        self.distilbert.pooler = None  # BERTweet does not use a pooler
        # print(self.bertweet.config)
        
        self.linear = nn.Linear(self.distilbert.config.dim * 2, self.distilbert.config.dim)  #1
        self.out = nn.Linear(self.distilbert.config.dim, num_labels)

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs["input_ids"], kwargs["attention_mask"]

        # Debugging: Print shapes before passing into BERTweet
        # print(f"x_input_ids.shape: {x_input_ids.shape}")

        # print(f"x_atten_masks.shape: {x_atten_masks.shape}")

        # print(f"x_input_ids.device: {x_input_ids.device}")  
        # print(f"x_atten_masks.device: {x_atten_masks.device}")
        
        # max_len = x_input_ids.size(1)  # length of the sequence
        # print(f"Max sequence length: {max_len}")
        # print(x_input_ids[0])
        # print(x_atten_masks[0])
        # print(x_input_ids)
        # # # print(x_input_ids.device, x_atten_masks.device)
        # # # print(type(x_input_ids), type(x_atten_masks))
        # print(x_atten_masks)
        # Ensure tensors are moved to the same device
        # assert not torch.isnan(x_input_ids).any(), "NaN found in input_ids"
        # assert not torch.isnan(x_atten_masks).any(), "NaN found in attention_mask"

        device = x_input_ids.device
        # # print(device)
        # print(device)


        # raise Exception

        last_hidden = self.distilbert(x_input_ids, x_atten_masks).last_hidden_state
        # print("SB"*10)
        
        # Use correct eos_token_id for Roberta
        eos_token_id = 102
        
        eos_token_ind = x_input_ids.eq(eos_token_id).nonzero()

        # Debugging: Check tensor lengths
        # print(f"len(eos_token_ind): {len(eos_token_ind)}")
        # print(f"len(x_input_ids): {len(x_input_ids)}, 3 * len(x_input_ids): {3 * len(x_input_ids)}")

        # Fix assertion to prevent crashes
        assert len(eos_token_ind) == 2 * len(x_input_ids), "Mismatch in EOS token count!"

        b_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if i % 2 == 0]
        e_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if (i + 1) % 2 == 0]
        
        x_atten_clone = x_atten_masks.clone().detach()

        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[: begin + 1] = 0, 0  # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0  # <s> --> 0; 3rd </s> --> 0

        # Convert attention masks to float and ensure they are on the same device
        txt_l = x_atten_masks.sum(1).to(device)
        topic_l = x_atten_clone.sum(1).to(device)
        txt_vec = x_atten_masks.float().to(device)
        topic_vec = x_atten_clone.float().to(device)

        txt_mean = torch.einsum("blh,bl->bh", last_hidden, txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum("blh,bl->bh", last_hidden, topic_vec) / topic_l.unsqueeze(1)

        # Concatenate sentence representations
        cat = torch.cat((txt_mean, topic_mean), dim=1)

        # Apply dropout, activation, and final classification layers
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out


# class bert_classifier(nn.Module):
#     def __init__(self, gen, model_name="bert-base-uncased", num_classes=3):
#         super(bert_classifier, self).__init__()
#         # Load the pre-trained BERT model
#         self.bert = BertModel.from_pretrained(model_name)
#         # Add two fully connected layers
#         self.fc1 = nn.Linear(self.bert.config.hidden_size, 128)  # First FC layer
#         self.relu = nn.ReLU()  # Activation
#         self.fc2 = nn.Linear(128, num_classes)  # Second FC layer for 3 output classes
#         self.dropout = nn.Dropout(0.1) if gen==0 else nn.Dropout(0.7)
#     def forward(self, **kwargs):

#         input_ids, attention_mask, token_type_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
#         # Pass inputs through BERT
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids
#         )
#         # Get the [CLS] token embedding
#         cls_output = outputs.last_hidden_state[:, 0, :]  # Using the raw [CLS] token embedding
#         # Pass through fully connected layers
#         x = self.dropout(cls_output)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
    
# BART
# class Encoder(BartPreTrainedModel):
    
#     def __init__(self, config: BartConfig):
        
#         super().__init__(config)

#         padding_idx, vocab_size = config.pad_token_id, config.vocab_size
#         self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
#         self.encoder = BartEncoder(config, self.shared)

#     def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False):

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         encoder_outputs = self.encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         return encoder_outputs


# Original bart classifier

class bart_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(bart_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
        self.bart = BartModel.from_pretrained("facebook/bart-large-mnli")
        self.encoder = self.bart.get_encoder()
        self.bart.pooler = None
        self.linear = nn.Linear(self.bart.config.hidden_size*2, self.bart.config.hidden_size) #1
        self.out = nn.Linear(self.bart.config.hidden_size, num_labels)

        # # Dropout
        # self.dropout = nn.Dropout(dropout) if gen == 0 else nn.Dropout(dropoutrest)
        # self.relu = nn.ReLU()
        
        # self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
        # self.encoder = BartModel.from_pretrained("facebook/bart-large-mnli").get_encoder()
        # # self.encoder.pooler = None
        # self.encoder.pooler = BartModel.from_pretrained("facebook/bart-large-mnli").pooler

        
        # # Linear layers
        # self.linear = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        # self.out = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        # print(x_input_ids[0])
        # print(x_atten_masks[0])
        # raise Exception
        # last_hidden = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks).last_hidden_state
        last_hidden = self.encoder(input_ids=x_input_ids, attention_mask=x_atten_masks).last_hidden_state
        
        eos_token_ind = x_input_ids.eq(self.config.eos_token_id).nonzero() # tensor([[0,4],[0,5],[0,11],[1,2],[1,3],[1,6]...])
        
        # print("x_input_ids:",x_input_ids,x_input_ids.size())
        # print("x_atten_masks:",x_atten_masks,x_atten_masks.size())

        # print("len(eos_token_ind):",len(eos_token_ind))
        # print("len(x_input_ids):",len(x_input_ids),3*len(x_input_ids))
        # print(bk)
        assert len(eos_token_ind) == 3*len(kwargs['input_ids'])
        b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i%3==0]
        e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i+1)%3==0]
        x_atten_clone = x_atten_masks.clone().detach()

        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[:begin+2] = 0, 0 # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0 # <s> --> 0; 3rd </s> --> 0
        
        txt_l = x_atten_masks.sum(1).to('cuda')
        topic_l = x_atten_clone.sum(1).to('cuda')
        txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = x_atten_clone.type(torch.FloatTensor).to('cuda')
        txt_mean = torch.einsum('blh,bl->bh', last_hidden, txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden, topic_vec) / topic_l.unsqueeze(1)

        cat = torch.cat((txt_mean, topic_mean), dim=1) #1
        # cat = torch.cat((txt_mean, topic_mean, txt_mean - topic_mean, txt_mean * topic_mean), dim=1)
        
        
        # raise Exception
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out
        

    
# class Encoder(BartPreTrainedModel):
    
#     def __init__(self, config: BartConfig):
        
#         super().__init__(config)

#         padding_idx, vocab_size = config.pad_token_id, config.vocab_size
#         self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
#         self.encoder = BartEncoder(config, self.shared)

#     def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False):

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         encoder_outputs = self.encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         return encoder_outputs

# class bart_mnli_encoder_classifier(nn.Module):

#     def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

#         super(bart_mnli_encoder_classifier, self).__init__()
        
#         self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
#         self.relu = nn.ReLU()
        
#         self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
#         self.bart = Encoder.from_pretrained("facebook/bart-large-mnli")
#         self.bart.pooler = None
#         # self.linear = nn.Linear(self.bart.config.hidden_size*2, self.bart.config.hidden_size)
#         self.linear = nn.Linear(self.bart.config.hidden_size, self.bart.config.hidden_size)
#         self.out = nn.Linear(self.bart.config.hidden_size, num_labels)
        
#     def forward(self, **kwargs):
        
#         x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
#         last_hidden = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks)

#         cls_hidden = last_hidden[0][:, 0, :]
#         query = self.dropout(cls_hidden)
#         linear = self.relu(self.linear(query))
#         out = self.out(linear)
#         return out



# class bart_classifier(nn.Module):

#     def __init__(self, num_labels, model_select, gen, dropout):

#         super(bart_classifier, self).__init__()
        
#         self.dropout = nn.Dropout(0.1) if gen==0 else nn.Dropout(dropout)
#         self.relu = nn.ReLU()
        
#         self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
#         self.bart = Encoder.from_pretrained("facebook/bart-large-mnli")
#         self.linear = nn.Linear(self.bart.config.hidden_size, self.bart.config.hidden_size)
#         self.out = nn.Linear(self.bart.config.hidden_size, num_labels)
        
#     def forward(self, **kwargs):
        
#         x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
#         last_hidden = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks)
        
#         hidden_states = last_hidden[0] 
#         eos_mask = x_input_ids.eq(self.config.eos_token_id)

#         if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
#             raise ValueError("All examples must have the same number of <eos> tokens.")
#         query = hidden_states[eos_mask,:].view(hidden_states.size(0), -1, hidden_states.size(-1))[:,-1,:]

#         query = self.dropout(query)
#         linear = self.relu(self.linear(query))
#         out = self.out(linear)
        
#         return out







# simple bahdanau attention
# class bart_classifier(nn.Module):
#     def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):
#         super(bart_classifier, self).__init__()

#         self.dropout = nn.Dropout(dropout) if gen == 0 else nn.Dropout(dropoutrest)
#         self.relu = nn.ReLU()

#         # Load Bart model and its configurations
#         self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
#         self.bart = BartModel.from_pretrained("facebook/bart-large-mnli")
#         self.encoder = self.bart.get_encoder()
#         self.bart.pooler = None

#         # Linear layers for classification
#         self.linear = nn.Linear(self.bart.config.hidden_size * 5, self.bart.config.hidden_size)
#         self.out = nn.Linear(self.bart.config.hidden_size, num_labels)

#         # Initialize Bahdanau Attention
#         self.attention = BahdanauAttention(
#             encoder_hidden_dim=self.bart.config.hidden_size,
#             decoder_hidden_dim=self.bart.config.hidden_size,
#             attention_dim=128
#         )

#     def forward(self, **kwargs):
#         x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
#         last_hidden = self.encoder(input_ids=x_input_ids, attention_mask=x_atten_masks).last_hidden_state

#         # Identify the positions of EOS tokens
#         eos_token_ind = x_input_ids.eq(self.config.eos_token_id).nonzero()
#         assert len(eos_token_ind) == 3 * len(kwargs['input_ids'])

#         b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i % 3 == 0]
#         e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i + 1) % 3 == 0]
#         x_atten_clone = x_atten_masks.clone().detach()

#         # Mask out irrelevant sections
#         for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
#             att[begin:], att2[:begin + 2] = 0, 0
#             att[0], att2[end] = 0, 0

#         # Isolate text and target representations
#         text_mask = x_atten_masks
#         target_mask = x_atten_clone

#         text_l = text_mask.sum(1).to('cuda')
#         target_l = target_mask.sum(1).to('cuda')
#         text_vec = text_mask.type(torch.FloatTensor).to('cuda')
#         target_vec = target_mask.type(torch.FloatTensor).to('cuda')

#         # Weighted mean of text and target representations
#         text_rep = torch.einsum('blh,bl->bh', last_hidden, text_vec) / text_l.unsqueeze(1)
#         target_rep = torch.einsum('blh,bl->bh', last_hidden, target_vec) / target_l.unsqueeze(1)

#         # Apply Bahdanau Attention between text and target
#         context_vector, _ = self.attention(text_rep.unsqueeze(1), target_rep)

#         # Concatenate context vector with text representation
#         cat = torch.cat((text_rep, target_rep, text_rep-target_rep, text_rep * target_rep,  context_vector), dim=1)
#         query = self.dropout(cat)
#         linear = self.relu(self.linear(query))
#         out = self.out(linear)

#         return out



# import torch
# import torch.nn as nn
# from transformers import BartModel, BartConfig

# class BahdanauAttention(nn.Module):
#     def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
#         super(BahdanauAttention, self).__init__()
#         self.encoder_attn = nn.Linear(encoder_hidden_dim, attention_dim)
#         self.decoder_attn = nn.Linear(decoder_hidden_dim, attention_dim)
#         self.v = nn.Linear(attention_dim, 1)

#     def forward(self, encoder_outputs, decoder_hidden):
#         encoder_energy = self.encoder_attn(encoder_outputs)  # (batch_size, seq_len, attention_dim)
#         decoder_energy = self.decoder_attn(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)
#         energy = torch.tanh(encoder_energy + decoder_energy)  # (batch_size, seq_len, attention_dim)
#         attention_weights = torch.softmax(self.v(energy).squeeze(-1), dim=1)  # (batch_size, seq_len)
#         context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, encoder_hidden_dim)
#         context_vector = context_vector.squeeze(1)  # (batch_size, encoder_hidden_dim)
#         return context_vector, attention_weights

# class bart_classifier(nn.Module):
#     def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):
#         super(bart_classifier, self).__init__()

#         self.dropout = nn.Dropout(dropout) if gen == 0 else nn.Dropout(dropoutrest)
#         self.relu = nn.ReLU()

#         self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
#         self.bart = BartModel.from_pretrained("facebook/bart-large-mnli")
#         self.encoder = self.bart.get_encoder()
#         self.bart.pooler = None
        
#         self.linear = nn.Linear(self.bart.config.hidden_size * 4, self.bart.config.hidden_size)
#         self.out = nn.Linear(self.bart.config.hidden_size, num_labels)

#         # Initialize Bahdanau Attention
#         self.attention = BahdanauAttention(
#             encoder_hidden_dim=self.bart.config.hidden_size,
#             decoder_hidden_dim=self.bart.config.hidden_size,
#             attention_dim=128
#         )

#         # Cross-attention layers for text and target
#         self.text_to_target_attention = nn.MultiheadAttention(embed_dim=self.bart.config.hidden_size, num_heads=8)
#         self.target_to_text_attention = nn.MultiheadAttention(embed_dim=self.bart.config.hidden_size, num_heads=8)

#     def forward(self, **kwargs):
#         x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        
#         # Extract encoder outputs
#         encoder_outputs = self.encoder(input_ids=x_input_ids, attention_mask=x_atten_masks, output_hidden_states=True).last_hidden_state

#         # Separate text and target parts using masks
#         sep_indices = (x_input_ids == 2).nonzero(as_tuple=False)  # Find indices of [SEP] tokens (value 2)
#         batch_size = x_input_ids.size(0)

#         text_masks = torch.zeros_like(x_atten_masks)
#         target_masks = torch.zeros_like(x_atten_masks)

#         for i in range(batch_size):
#             sep_batch = sep_indices[sep_indices[:, 0] == i][:, 1]
#             assert len(sep_batch) >= 3, f"Batch index {i} must have at least three [SEP] tokens."
            
#             b_sep, e_sep = sep_batch[0], sep_batch[2]

#             text_masks[i, : b_sep + 1] = 1
#             target_masks[i, e_sep + 1 :] = 1

#         # Ensure no division by zero
#         text_l = text_masks.sum(1).to('cuda')
#         target_l = target_masks.sum(1).to('cuda')

#         text_l = torch.clamp(text_l, min=1)  # Prevent division by zero
#         target_l = torch.clamp(target_l, min=1)  # Prevent division by zero

#         text_vec = text_masks.type(torch.FloatTensor).to('cuda')
#         target_vec = target_masks.type(torch.FloatTensor).to('cuda')

#         # Safe mean calculation, avoiding division by zero
#         text_mean = torch.einsum('blh,bl->bh', encoder_outputs, text_vec) / text_l.unsqueeze(1)
#         target_mean = torch.einsum('blh,bl->bh', encoder_outputs, target_vec) / target_l.unsqueeze(1)

#         # Cross-Attention: Text attends to Target
#         text_attends_target, _ = self.text_to_target_attention(
#             query=encoder_outputs.permute(1, 0, 2),  # Query comes from the text part of the encoder
#             key=encoder_outputs.permute(1, 0, 2),   # Key and Value come from the target part of the encoder
#             value=encoder_outputs.permute(1, 0, 2)
#         )

#         # Cross-Attention: Target attends to Text
#         target_attends_text, _ = self.target_to_text_attention(
#             query=encoder_outputs.permute(1, 0, 2),  # Query comes from the target part
#             key=encoder_outputs.permute(1, 0, 2),   # Key and Value come from the text part
#             value=encoder_outputs.permute(1, 0, 2)
#         )

#         text_attends_target = text_attends_target.permute(1, 0, 2)
#         target_attends_text = target_attends_text.permute(1, 0, 2)

#         context_text, _ = self.attention(text_attends_target, text_mean)
#         context_target, _ = self.attention(target_attends_text, target_mean)

#         # Concatenate the contexts from both text and target
#         cat = torch.cat((context_text, context_target, context_text - context_target, context_text * context_target), dim=1)
        
#         # Apply dropout and linear transformation
#         query = self.dropout(cat)
#         linear = self.relu(self.linear(query))

#         # Logits (classification output)
#         out = self.out(linear)

#         # Ensure no NaN values in output logits
#         out = torch.nan_to_num(out, nan=0.0)  # Replace NaN logits with 0

#         return out





# class bart_classifier(nn.Module):
#     def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):
#         super(bart_classifier, self).__init__()

#         self.dropout = nn.Dropout(dropout) if gen == 0 else nn.Dropout(dropoutrest)
#         self.relu = nn.ReLU()

#         # Load Bart model and its configurations
#         self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
#         self.bart = BartModel.from_pretrained("facebook/bart-large-mnli")
#         self.encoder = self.bart.get_encoder()
#         self.bart.pooler = None

#         # Linear layers for classification
#         self.linear = nn.Linear(self.bart.config.hidden_size * 7, self.bart.config.hidden_size)
#         self.out = nn.Linear(self.bart.config.hidden_size, num_labels)

#         # Initialize Bahdanau Attention
#         self.attention = BahdanauAttention(
#             encoder_hidden_dim=self.bart.config.hidden_size,
#             decoder_hidden_dim=self.bart.config.hidden_size,
#             attention_dim=128
#         )

#         # Attention mechanisms
#         self.cross_attention_text_to_target = nn.MultiheadAttention(embed_dim=self.bart.config.hidden_size, num_heads=8)
#         self.cross_attention_target_to_text = nn.MultiheadAttention(embed_dim=self.bart.config.hidden_size, num_heads=8)

#     def forward(self, **kwargs):
#         x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
#         last_hidden = self.encoder(input_ids=x_input_ids, attention_mask=x_atten_masks).last_hidden_state

#         # Identify the positions of EOS tokens
#         eos_token_ind = x_input_ids.eq(self.config.eos_token_id).nonzero()
#         assert len(eos_token_ind) == 3 * len(kwargs['input_ids'])

#         b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i % 3 == 0]
#         e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i + 1) % 3 == 0]
#         x_atten_clone = x_atten_masks.clone().detach()

#         # Mask out irrelevant sections
#         for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
#             att[begin:], att2[:begin + 2] = 0, 0
#             att[0], att2[end] = 0, 0

#         # Isolate text and target representations
#         text_mask = x_atten_masks
#         target_mask = x_atten_clone

#         text_l = text_mask.sum(1).to('cuda')
#         target_l = target_mask.sum(1).to('cuda')
#         text_vec = text_mask.type(torch.FloatTensor).to('cuda')
#         target_vec = target_mask.type(torch.FloatTensor).to('cuda')

#         # Weighted mean of text and target representations
#         text_rep = torch.einsum('blh,bl->bh', last_hidden, text_vec) / text_l.unsqueeze(1)
#         target_rep = torch.einsum('blh,bl->bh', last_hidden, target_vec) / target_l.unsqueeze(1)

        

#         # Cross-attention: Text to Target
#         text_as_query = text_rep.unsqueeze(0)  # Shape: (1, batch_size, hidden_dim)
#         target_as_key_value = target_rep.unsqueeze(0)  # Shape: (1, batch_size, hidden_dim)
#         cross_text_to_target, _ = self.cross_attention_text_to_target(
#             query=text_as_query, key=target_as_key_value, value=target_as_key_value
#         )

#         # Cross-attention: Target to Text
#         target_as_query = target_rep.unsqueeze(0)  # Shape: (1, batch_size, hidden_dim)
#         text_as_key_value = text_rep.unsqueeze(0)  # Shape: (1, batch_size, hidden_dim)
#         cross_target_to_text, _ = self.cross_attention_target_to_text(
#             query=target_as_query, key=text_as_key_value, value=text_as_key_value
#         )

#         # context_text, _ = self.attention(cross_text_to_target, text_rep)
#         context_vector, _ = self.attention(text_rep.unsqueeze(1), target_rep)
        
#         cross_target_to_text = cross_target_to_text.squeeze(0)
#         cross_text_to_target = cross_text_to_target.squeeze(0)


        

#         # Concatenate all representations
#         cat = torch.cat(
#             (text_rep, target_rep, text_rep-target_rep, text_rep*target_rep, cross_text_to_target, cross_target_to_text, context_vector), dim=1
#         )
#         query = self.dropout(cat)
#         linear = self.relu(self.linear(query))
#         out = self.out(linear)

#         return out