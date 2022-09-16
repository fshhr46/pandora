import logging

import torch

logger = logging.getLogger(__name__)


def build_output(logits, id2label):
    y_hat = logits.argmax(0).item()
    y_softmax = torch.softmax(logits, 0)
    predicted_idx = y_hat

    named_softmax = {}
    for idx, prob in enumerate(y_softmax.tolist()):
        name = id2label[idx]
        named_softmax[name] = prob
    return {
        "class": id2label[predicted_idx],
        "probability": named_softmax[id2label[predicted_idx]],
        "softmax": named_softmax,
    }


def run_inference(mode: str, model, id2label, input_batch):
    """Predict the class (or classes) of the received text using the
    serialized transformers checkpoint.
    Args:
        input_batch (list): List of Text Tensors from the pre-process function is passed here

            input_ids_batch, attention_mask_batch, token_type_ids, labels, indexes
    Returns:
        list : It returns a list of the predicted value for the input text
    """
    # token_type_ids, labels are place holders
    input_ids_batch, attention_mask_batch, token_type_ids_batch, indexes = input_batch
    inputs = {"input_ids": input_ids_batch,
              "attention_mask": attention_mask_batch,
              "token_type_ids": token_type_ids_batch,
              }
    inferences = []
    # Handling inference for sequence_classification.
    if mode == "sequence_classification":
        outputs = model(**inputs)
        predictions = outputs[0]

        num_preds, num_classes = predictions.shape
        for i in range(num_preds):
            formatted_output = build_output(predictions[i], id2label)
            inferences.append(formatted_output)
    # Handling inference for token_classification.
    elif mode == "token_classification":
        raise NotImplementedError

    # if indexes is passed, merge results (column level inferencing)
    if indexes:
        # merge inferences output for column level prediction requests.
        start = 0
        merged_inferences = []
        for index in indexes:
            sub_inferences = inferences[start:index+1]
            if len(sub_inferences) == 1:
                merged_inferences.append(sub_inferences[0])
            else:
                names = [inf["class"] for inf in sub_inferences]
                merged_output = merge_inferences(names)
                merged_inferences.append(merged_output)
            start = index
        inferences = merged_inferences
    return inferences


def merge_inferences(inferences):
    # TODO: replace naive implementation, simply take the max count.
    max_count_name = max(set(inferences), key=inferences.count)
    return {
        "class": max_count_name,
        "probability": 1.0 * inferences.count(max_count_name) / len(inferences),
        "softmax": [],
    }
