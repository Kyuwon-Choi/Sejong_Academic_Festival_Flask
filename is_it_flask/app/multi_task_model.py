import torch
from transformers import BertModel
import os

class MultiTaskBertModel(torch.nn.Module):
    def __init__(self, num_category_labels, num_function_labels, num_usage_base_labels):
        super(MultiTaskBertModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.category_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_category_labels)
        self.function_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_function_labels)
        self.usage_base_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_usage_base_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # BERT's pooled output

        category_logits = self.category_classifier(pooled_output)
        function_logits = self.function_classifier(pooled_output)
        usage_base_logits = self.usage_base_classifier(pooled_output)

        return category_logits, function_logits, usage_base_logits
    
    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        # BERT 모델 저장
        self.bert.save_pretrained(save_directory)
        # 분류기 저장
        torch.save(self.category_classifier.state_dict(), f'{save_directory}/category_classifier.bin')
        torch.save(self.function_classifier.state_dict(), f'{save_directory}/function_classifier.bin')
        torch.save(self.usage_base_classifier.state_dict(), f'{save_directory}/usage_base_classifier.bin')
    
    @classmethod
    def from_pretrained(cls, load_directory, num_category_labels, num_function_labels, num_usage_base_labels):
        model = cls(num_category_labels, num_function_labels, num_usage_base_labels)
        # BERT 모델 로드
        model.bert = BertModel.from_pretrained(load_directory)
        # 분류기 로드
        model.category_classifier.load_state_dict(torch.load(f'{load_directory}/category_classifier.bin'))
        model.function_classifier.load_state_dict(torch.load(f'{load_directory}/function_classifier.bin'))
        model.usage_base_classifier.load_state_dict(torch.load(f'{load_directory}/usage_base_classifier.bin'))
        return model
