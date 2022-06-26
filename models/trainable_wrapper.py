import torch

from transformers import BertTokenizer, BertModel, BertConfig

from models.GroupSparseActivation import GroupSparseActivation


class TrainableWrapper(torch.nn.Module):
    def __init__(self, config, logger):
        super(TrainableWrapper, self).__init__()

        self.config = config
        self.logger = logger
        max_seq_len_dict = {"imdb": 512, "sst2": 64, "ag-news": 256}
        self.max_seq_len = max_seq_len_dict[config.dataset]
        self.seq_len_cache = []
        num_classes = {"imdb": 2, "sst2": 2, "ag-news": 4}
        self.num_classes = num_classes[config.dataset]

        self.device = torch.device("cuda")

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", use_fast=True
        )

        bert_config = BertConfig.from_pretrained("bert-base-uncased")
        bert_config.num_hidden_layers = 1
        self.bert_embedding = BertModel.from_pretrained(
            "bert-base-uncased", 
            config=bert_config
        )
        #for p in self.bert_embedding.parameters():
        #    p.requires_grad = False

        self.bert_embedding.to(self.device)


        if config.gs:
            self.activation = GroupSparseActivation(2, 16, 2, 2, 2)
        else:
            self.activation = torch.nn.ReLU()

        self.lin = torch.nn.Linear(768, 128)
        self.clf = torch.nn.Linear(128, self.num_classes)
        self.softmax = torch.nn.Softmax(-1)
        self.to(self.device)


    def forward(self, text):

        tokens = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        attention_mask = tokens["attention_mask"].to(self.device)

        out = self.bert_embedding(
            input_ids=tokens["input_ids"].to(self.device),
            attention_mask=attention_mask,
            token_type_ids=(
                tokens["token_type_ids"].to(self.device)
                if "token_type_ids" in tokens
                else None
            ),
            output_hidden_states=False,
            output_attentions=False
        )

        out = self.lin(out["pooler_output"])
        out = self.activation(out)
        out = self.clf(out)
        return self.softmax(out)


