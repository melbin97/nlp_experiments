#!/usr/bin/env python3
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

model=BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer=BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

f=open('chatbot.txt','r')
para=f.read()
para=para.lower()
question=input()

encoding=tokenizer.encode_plus(text=question,text_pair=para,add_special=True)

inputs=encoding['input_ids']
sentence_embedding=encoding['token_type_ids']
tokens=tokenizer.convert_ids_to_tokens(inputs)

startScores,endScores=model(input_ids=torch.tensor([inputs]),token_type_ids=torch.tensor([sentence_embedding]))

startIndex=torch.argmax(startScores)
endIndex=torch.argmax(endScores)
answer=' '.join(tokens[startIndex:endIndex+1])
print(answer)
