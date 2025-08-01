import torch
from transformers import AdamW
from utils import modeling
from transformers import BartForSequenceClassification
from torchinfo import summary
#from transformers import AutoModelForSequenceClassification, AutoTokenizer


def model_setup(num_labels, model_select, device, config, gen, dropout, dropoutrest):
    
    print("current dropout is: ", dropout, dropoutrest)
    if model_select == 'Bert':
        print(100*"#")
        print("using Bert")
        print(100*"#")
        model = modeling.bert_classifier(num_labels, gen, dropout, dropoutrest).to(device)
        ## model = modeling.bert_classifier(gen).to(device)
        for n, p in model.named_parameters():
            if "bert.embeddings" in n:
                p.requires_grad = False
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]
        
    elif model_select == 'ModernBert':
        print(100*"#")
        print("using ModernBert")
        print(100*"#")

        # model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli", num_labels = 3).to(device)

        # model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli', num_labels=3).to(device)

        model = modeling.modern_bert_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)

        # model = modeling.bart_mnli_encoder_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)

        for n, p in model.named_parameters():
            if "mbert.embeddings" in n:
                p.requires_grad = False
            
                
        # raise Exception
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('mbert.layers')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ] 

    elif model_select == 'Bertweet': #Bertweet
        print(100*"#")
        print("using BERTweet")
        print(100*"#")
        
        # Initialize the BERTweet model
        model = modeling.bertweet_classifier(num_labels, gen, dropout, dropoutrest).to(device)
        # print(model)

        # Freeze BERTweet's embedding layer
        for n, p in model.named_parameters():
            if "bertweet.embeddings" in n:
                p.requires_grad = False
            

        # raise Exception
        # Set up optimizer groups
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bertweet.encoder')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
        ]
    elif model_select == 'Roberta': 
        print(100*"#")
        print("using RoBERTa")
        print(100*"#")
        
        # Initialize the BERTweet model
        model = modeling.roberta_classifier(num_labels, gen, dropout, dropoutrest).to(device)
        # print(model)

        # Freeze BERTweet's embedding layer
        for n, p in model.named_parameters():
            if "roberta.embeddings" in n:
                p.requires_grad = False
        

        # raise Exception
        # Set up optimizer groups
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('roberta.encoder')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
        ]
    elif model_select == 'Distilbert': 
            print(100*"#")
            print("using DistilBERT")
            print(100*"#")
            
            # Initialize the BERTweet model
            model = modeling.distilbert_classifier(num_labels, gen, dropout, dropoutrest).to(device)
            # print(model)

            # Freeze BERTweet's embedding layer
            for n, p in model.named_parameters():
                if "distilbert.embeddings" in n:
                    p.requires_grad = False
                
            

            # raise Exception
            # Set up optimizer groups
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if n.startswith('distilbert.transformer')] , 'lr': float(config['bert_lr'])},
                {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
                {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]
    elif model_select == 'Deberta': 
            print(100*"#")
            print("using Deberta")
            print(100*"#")
            
            # Initialize the BERTweet model
            model = modeling.deberta_classifier(num_labels, gen, dropout, dropoutrest).to(device)
            # print(model)

            # Freeze BERTweet's embedding layer
            for n, p in model.named_parameters():
                # print(n, p.requires_grad)
                if "deberta.embeddings" in n:
                    p.requires_grad = False
            
            

            # raise Exception
            # Set up optimizer groups
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if n.startswith('deberta.encoder.layer')] , 'lr': float(config['bert_lr'])},
                {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
                {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]

    elif model_select == 'Bart':
        print(100*"#")
        print("using Bart")
        print(100*"#")

        # model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli", num_labels = 3).to(device)

        # model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli', num_labels=3).to(device)

        model = modeling.bart_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)

        # model = modeling.bart_mnli_encoder_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)

        for n, p in model.named_parameters():
            if "bart.shared.weight" in n or "bart.encoder.embed" in n:
                p.requires_grad = False
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bart.encoder.layer')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]
    
    optimizer = AdamW(optimizer_grouped_parameters)
    # optimizer = AdamW(model.parameters(), lr=5e-5)    
    return model, optimizer

    


def model_preds(loader, model, device, loss_function):
    
    preds = [] 
    valtest_loss = []   
    for b_id, sample_batch in enumerate(loader):

        dict_batch = batch_fn(sample_batch)
        inputs = {k: v.to(device) for k, v in dict_batch.items()}
        
        outputs = model(**inputs)
        preds.append(outputs)
        loss = loss_function(outputs, inputs['gt_label'])

        # outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['gt_label'])
        # loss = outputs.loss
        # preds.append(outputs.logits)


        valtest_loss.append(loss.item())

    return torch.cat(preds, 0), valtest_loss


def batch_fn(sample_batch):
    
    dict_batch = {}
    dict_batch['input_ids'] = sample_batch[0]
    dict_batch['attention_mask'] = sample_batch[1]
    dict_batch['gt_label'] = sample_batch[-1]
    if len(sample_batch) > 3:
        dict_batch['token_type_ids'] = sample_batch[-2]
    
    return dict_batch