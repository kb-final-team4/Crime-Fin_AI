import torch
import torch.nn.functional as F
import os

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from kobert_transformers.utils import get_tokenizer
# from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert_tokenizer import KoBERTTokenizer
# import gluonnlp as nlp


def classify(config) -> list:
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )
    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']

    lines = config.lines

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        print(train_config['batch_size'])
        print(train_config['gpu_id'])
        print(train_config['pretrained_model_name'])
        print(train_config['model_fn'])
        print(os.path.exists(train_config['model_fn']))

        tokenizer = KoBERTTokenizer.from_pretrained(
            train_config['pretrained_model_name'])

        model_loader = BertForSequenceClassification

        model = model_loader.from_pretrained(
            train_config['pretrained_model_name'],
            num_labels=len(index_to_label)
        )
        model.load_state_dict(bert_best)  # type: ignore

        if train_config['gpu_id'] >= 0:
            model.cuda(train_config['gpu_id'])  # type: ignore

        device = next(model.parameters()).device  # type: ignore

        model.eval()  # type: ignore

        y_hats = []
        for idx in range(0, len(lines), config.batch_size):
            mini_batch = tokenizer(
                lines[idx:idx + config.batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            x = mini_batch['input_ids']
            x = x.to(device)
            mask = mini_batch['attention_mask']
            mask = mask.to(device)

            # Take feed-forward
            y_hat = F.softmax(
                model(x, attention_mask=mask).logits, dim=-1)  # type: ignore

            y_hats += [y_hat]
        # Concatenate the mini-batch wise result
        y_hats = torch.cat(y_hats, dim=0)
        # |y_hats| = (len(lines), n_classes)

        probs, indice = y_hats.cpu().topk(config.top_k)
        # |indice| = (len(lines), top_k)

        result = []
        for i, line in enumerate(lines):
            # classification probability, 분류한 텍스트를 담아 반환.
            row = [float(probs[i][0]), index_to_label[int(indice[i][0])], line]
            result.append(row)
        return result
