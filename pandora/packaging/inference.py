import json
import logging

import torch

from captum.attr import LayerIntegratedGradients

logger = logging.getLogger(__name__)


def _format_output(logits, id2label):
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


def format_outputs(inferences, id2label):
    return [_format_output(inference, id2label) for inference in inferences]


def run_inference(input_batch, mode: str, model):
    """Predict the class (or classes) of the received text using the
    serialized transformers checkpoint.
    Args:
        input_batch (list): List of Text Tensors from the pre-process function is passed here

            input_ids_batch, attention_mask_batch, token_type_ids, labels, indexes
    Returns:
        list : It returns a list of the predicted value (list of logits) for the input text
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
        logits_batch = outputs[0]

        num_preds, num_classes = logits_batch.shape
        for i in range(num_preds):
            inference = logits_batch[i]
            inferences.append(inference)
    else:
        raise NotImplementedError
    return inferences


def merge_outputs(formated_outputs, indexes):
    # merge inferences output for column level prediction requests.
    start = 0
    merged_outputs = []
    for index in indexes:
        sub_inferences = formated_outputs[start:index+1]
        if len(sub_inferences) == 1:
            merged_outputs.append(sub_inferences[0])
        else:
            names = [inf["class"] for inf in sub_inferences]
            merged_output = _merge_outputs(names)
            merged_outputs.append(merged_output)
        start = index
    return merged_outputs


def _merge_outputs(formated_outputs):
    # TODO: replace naive implementation, simply take the max count.
    # max_count_name = max(set(formated_outputs), key=formated_outputs.count)
    # return {
    #     "class": max_count_name,
    #     "probability": 1.0 * formated_outputs.count(max_count_name) / len(formated_outputs),
    #     "softmax": [],
    # }
    raise NotImplementedError


def run_get_insights(
        # configs
        mode: str,
        embedding_name: str,
        captum_explanation: bool,
        # model related
        model,
        tokenizer,
        # device
        device,
        # Input related
        # Original API def
        input_batch,
        target):
    """This function initialize and calls the layer integrated gradient to get word importance
    of the input text if captum explanation has been selected through setup_config
    Args:
        input_batch (int): Batches of tokens IDs of text
        text (str): The Text specified in the input request
        target (int): The Target can be set to any acceptable label under the user's discretion.
    Returns:
        (list): Returns a list of importances and words.
    """

    input_ids_batch, attention_mask_batch, token_type_ids_batch, indexes = input_batch
    if captum_explanation:
        embedding_layer = getattr(model, embedding_name)
        embeddings = embedding_layer.embeddings
        lig = LayerIntegratedGradients(captum_sequence_forward, embeddings)
    else:
        logger.warning(
            "Captum Explanation is not chosen and will not be available")

    batch_size, _ = input_ids_batch.shape

    # TODO: Fix construct_input_ref.
    # Currently construct_input_ref is adding [CLS] and [SEP] token to
    # the tensor but our feature extraction doesn't do that.
    # input_ids, ref_input_ids, attention_mask = construct_input_ref(
    #     input_ids_batch, tokenizer, device
    # )
    ref_input_ids_batch = torch.tensor(
        [tokenizer.pad_token_id] * input_ids_batch.numel(),
        device=device).reshape(input_ids_batch.shape)

    if mode == "sequence_classification":
        attributions, delta = lig.attribute(
            inputs=input_ids_batch,
            baselines=ref_input_ids_batch,
            target=target,
            additional_forward_args=(
                attention_mask_batch, token_type_ids_batch, indexes, mode, model),
            return_convergence_delta=True,
        )
        attributions_sum = summarize_attributions(attributions)
    else:
        raise NotImplementedError

    responses = []
    for i in range(batch_size):
        response = {}
        all_tokens = get_word_token(input_ids_batch, tokenizer, i)
        response["words"] = all_tokens
        response["importances"] = attributions_sum[i].tolist()
        response["delta"] = delta[i].tolist()
        responses.append(response)
    return responses


def construct_input_ref(input_ids, tokenizer, device):
    """For a given text, this function creates token id, reference id and
    attention mask based on encode which is faster for captum insights
    Args:
        text (str): The text specified in the input request
        tokenizer (AutoTokenizer Class Object): To word tokenize the input text
        device (cpu or gpu): Type of the Environment the server runs on.
    Returns:
        input_id(Tensor): It attributes to the tensor of the input tokenized words
        ref_input_ids(Tensor): Ref Input IDs are used as baseline for the attributions
        attention mask() :  The attention mask is a binary tensor indicating the position
         of the padded indices so that the model does not attend to them.
    """

    # text_ids = tokenizer.encode(text, add_special_tokens=False)
    text_ids = input_ids

    # construct input token ids
    logger.info("text_ids %s", text_ids)
    logger.info("[tokenizer.cls_token_id] %s", [tokenizer.cls_token_id])
    input_ids = [tokenizer.cls_token_id] + text_ids + [tokenizer.sep_token_id]
    logger.info("input_ids %s", input_ids)

    input_ids = torch.tensor([input_ids], device=device)
    # construct reference token ids
    ref_input_ids = (
        [tokenizer.cls_token_id]
        + [tokenizer.pad_token_id] * len(text_ids)
        + [tokenizer.sep_token_id]
    )
    ref_input_ids = torch.tensor([ref_input_ids], device=device)
    # construct attention mask
    attention_mask = torch.ones_like(input_ids)
    return input_ids, ref_input_ids, attention_mask


def get_word_token(input_ids, tokenizer, batch_id):
    """constructs word tokens from token id using the BERT's
    Auto Tokenizer
    Args:
        input_ids (list): Input IDs from construct_input_ref method
        tokenizer (class): The Auto Tokenizer Pre-Trained model object
    Returns:
        (list): Returns the word tokens
    """
    indices = input_ids[batch_id].detach().tolist()
    tokens = tokenizer.convert_ids_to_tokens(indices)
    # Remove unicode space character from BPE Tokeniser
    tokens = [token.replace("Ġ", "") for token in tokens]
    return tokens


def captum_sequence_forward(input_ids_batch, attention_mask_batch, token_type_ids_batch, indexes, mode: str, model):
    """This function is used to get the predictions from the model and this function
    can be used independent of the type of the BERT Task.
    Args:
        inputs (list): Input for Predictions
        attention_mask (list, optional): The attention mask is a binary tensor indicating the position
         of the padded indices so that the model does not attend to them, it defaults to None.
        position (int, optional): Position depends on the BERT Task.
        model ([type], optional): Name of the model, it defaults to None.
    Returns:
        list: Prediction Outcome
    """
    input_batch = (input_ids_batch, attention_mask_batch,
                   token_type_ids_batch, indexes)
    inferences = run_inference(input_batch, mode, model)
    return torch.stack(inferences)


def summarize_attributions(attributions):
    """Summarises the attribution across multiple runs
    Args:
        attributions ([list): attributions from the Layer Integrated Gradients
    Returns:
        list : Returns the attributions after normalizing them.
    """
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions
