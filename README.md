# NER test task

Named-entity recognition model.   
Input: Some text in english "Orshulevich Yurii Olegovich".   
Output: Lables for every word.

## Contents of the repository

In the notebook TestTaskNER.ipynb, the pipeline through which I experimented.    
In the model_inference.py file, the basis for the model's inference.   
I used the fine-tune of BERT models for token classification, conducted several experiments with different parameters and pretrained models.   
History of experiments and dashboards of training and metrics can be found [here](https://wandb.ai/ur_or/NER_test_task_ZERO).   

## Results

The best result was obtained with bert-base-uncased training, but it does not significantly exceed experiments with distillation models. In my opinion, the best model is distilbert-base-cased, despite its metrics being slightly lower than other models on the test set but better for the validation one. This can be explained by the fact that in the test set there are more sentences written entirely in capital letters, which spoils the prediction results. However, in real data, we will often see sentences as usual, and information about capital letters can help with entity recognition.
Result metrics:

|          **Model**          |     **Accuracy**    |   **F1 macro**   |
|:---------------------------:|:-------------------:|:----------------:|
|      bert-base-uncased      |        98,11 %      |      89,42 %     |
|   distilbert-base-uncased   |        97,97 %      |      88,83 %     |
|    distilbert-base-cased    |        97,95 %      |      88,77 %     |
