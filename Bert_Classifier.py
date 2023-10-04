import torch.nn as nn
import torch
import transformers
from transformers import BertModel, BertTokenizer
import joblib

class Bert_Classifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(Bert_Classifier, self).__init__()
        # Specify hidden size of BERT, hidden size of the classifier, and number of labels
        n_input = 768
        n_hidden = 50
        n_output = 2
        # Instantiate BERT model and tokenizer
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Add dense layers to perform the classification
        self.classifier = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )
        # Add possibility to freeze the BERT model
        # to avoid fine tuning BERT params (usually leads to worse results)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_ids, attention_mask):
        # Feed input data to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

    def predict(self, comment):
        inputs = self.tokenizer(comment, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        logits = self.forward(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=1)
        return predictions.numpy()[0]

    def predict_proba(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities.cpu().detach().numpy()[0]

# Création d'un objet Bert_Classifier
clf = Bert_Classifier()

# Enregistrement de l'objet avec joblib
joblib.dump(clf, "bert.pkl")
