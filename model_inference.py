from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List

import torch
import os


class NER_markuper():

    @staticmethod
    def align_label(word_ids: List[int], labels: List[str]) -> List[str]:

        previous_word_idx = None
        aligned_labels = []

        for word_id in word_ids:

            if word_id is None or word_id == previous_word_idx:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(labels[word_id])
            
            previous_word_idx = word_id

        return aligned_labels


    def __init__(self, path_to_model: os.PathLike):
        self.__tokenizer = AutoTokenizer.from_pretrained(path_to_model, use_auth_token=True)
        self.__model = AutoModelForTokenClassification.from_pretrained(path_to_model, use_auth_token=True)
        self.__model.eval()

        self.__id_to_label = {
            0: 'O',
            1: 'B-PER',
            2: 'I-PER',
            3: 'B-ORG',
            4: 'I-ORG',
            5: 'B-LOC',
            6: 'I-LOC',
            7: 'B-MISC',
            8: 'I-MISC',
        }

 
    def markup_entities(self, sentence: str) -> List[str]:

        sentence = sentence.lower()
        encoded_dict = self.__tokenizer.encode_plus(
            sentence.split(),
            padding='max_length',
            max_length = 256,
            truncation=True,
            return_tensors="pt",
            is_split_into_words=True
        )

        input_id = encoded_dict['input_ids'].to('cpu')
        mask = encoded_dict['attention_mask'].to('cpu')

        mask_for_labels = [1] * len(sentence.split())
        
        mask_for_labels = torch.tensor(NER_markuper.align_label(encoded_dict.word_ids(), mask_for_labels))
        
        with torch.no_grad():
            logits = self.__model(
                input_ids=input_id,
                attention_mask=mask,
                labels=None,
                return_dict=False
            )
            
        logits_clean = logits[0][0][mask_for_labels != -100]

        probabilities = torch.nn.functional.softmax(logits_clean, dim=1)
        predictions = probabilities.argmax(dim=1).tolist()
        
        prediction_label = [self.__id_to_label[i] for i in predictions]
        
        return prediction_label


def main():
    best_markuper = NER_markuper('model_outputs/finetuned-distilbert-base-uncased1')
    
    test_sentence = 'Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday.'
    pred_labels = best_markuper.markup_entities(test_sentence)

    print(test_sentence)
    print(pred_labels)
    print()

    for token, tag in zip(test_sentence.split(), pred_labels):
        print(token, '\t', tag)


if __name__ == '__main__':
    main()
