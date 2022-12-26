import json
import logging

import torch

from captum.attr import LayerIntegratedGradients

logger = logging.getLogger(__name__)


def _format_output(logits, sigmoids, id2label, doc_threshold):
    # When using DOC, sigmoids is already calculated
    y_hat = logits.argmax(0).item()
    y_softmax = torch.softmax(logits, 0).tolist()
    predicted_idx = y_hat

    if sigmoids is None:
        max_sigmoid = None
        y_sigmoids = None
    else:
        # Determine whether to reject the output and
        # output unknwon class.
        # Check if all labels' sigmoid are smaller than the threshold.
        max_sigmoid = sigmoids.max().item()

        # binary x_ent outputs
        y_sigmoids = sigmoids.tolist()

    # Determine whether to reject the output and
    # output unknwon class.
    # Check if all labels' sigmoid are smaller than the threshold.
    reject_output = max_sigmoid != None and max_sigmoid <= doc_threshold

    named_softmax = {}
    named_sigmoid = {}
    # TODO: Here' its a bit hacky:
    # len(y_softmax): actual number of labels used in training (Seen labels Only).
    # len(id2label):  All id2label mapping, including Seen and Unseen
    for idx, softmax in enumerate(y_softmax):
        label_name = id2label[idx]
        named_softmax[label_name] = softmax
        if y_sigmoids:
            named_sigmoid[label_name] = y_sigmoids[idx]

    formatted_output = {
        "class": id2label[predicted_idx],
        "target": predicted_idx,
        "probability": named_softmax[id2label[predicted_idx]],
        "softmax": named_softmax,
        "sigmoid": named_sigmoid,
        "rejected": reject_output,
        "max_sigmoid": max_sigmoid,
        "doc_threshold": doc_threshold,
    }
    return formatted_output


def format_outputs(logits_list, sigmoids_list, id2label, doc_threshold):
    # When sigmoids is passed, use it to determine unknown class.
    if sigmoids_list:
        return [_format_output(logits, sigmoids, id2label, doc_threshold=doc_threshold)
                for logits, sigmoids in zip(logits_list, sigmoids_list)]
    else:
        return [_format_output(logits, None, id2label, doc_threshold=doc_threshold)
                for logits in logits_list]


def run_inference(inputs, mode: str, model):
    """Predict the class (or classes) of the received text using the
    serialized transformers checkpoint.
    Args:
        input_batch (list): List of Text Tensors from the pre-process function is passed here

            input_ids_batch, attention_mask_batch, token_type_ids, labels, indexes
    Returns:
        list : It returns a list of the predicted value (list of logits) for the input text
    """
    # token_type_ids, labels are place holders
    logits_list = []
    sigmoids_list = None
    # Handling inference for sequence_classification.
    if mode == "sequence_classification":
        logits_batch, sigmoids_batch = model(**inputs)[:2]
        if sigmoids_batch is not None:
            sigmoids_list = []

        num_preds, num_classes = logits_batch.shape
        for i in range(num_preds):
            logits_list.append(logits_batch[i])
            if sigmoids_batch is not None:
                sigmoids_list.append(sigmoids_batch[i])
    else:
        raise NotImplementedError
    return logits_list, sigmoids_list


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
        # model related
        model,
        tokenizer,
        # device
        device,
        # Input related
        # Original API def
        input_batch,
        target: int,
        include_char_data: bool,
        n_steps=50):
    """This function initialize and calls the layer integrated gradient to get word importance
    of the input text if captum explanation has been selected through setup_config
    Args:
        input_batch (int): Batches of tokens IDs of text
        text (str): The Text specified in the input request
        target (int): The Target can be set to any acceptable label under the user's discretion.
    Returns:
        (list): Returns a list of importances and words.
    """

    if include_char_data:
        input_ids_batch, attention_mask_batch, token_type_ids_batch, \
            char_input_ids_batch, start_ids_batch, end_ids_batch, \
            indexes = input_batch
        additional_forward_args = (
            attention_mask_batch, token_type_ids_batch,
            char_input_ids_batch, start_ids_batch, end_ids_batch,
            indexes, mode, model)
        forward_func = captum_sequence_forward_char_bert
    else:
        input_ids_batch, attention_mask_batch, token_type_ids_batch, indexes = input_batch
        additional_forward_args = (
            attention_mask_batch, token_type_ids_batch, indexes, mode, model)
        forward_func = captum_sequence_forward
    embedding_layer = getattr(model, embedding_name)
    embeddings = embedding_layer.embeddings
    lig = LayerIntegratedGradients(
        forward_func=forward_func,
        layer=embeddings)
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
        # attributions: shape - batch_size, num_tokens, bert's num_hidden_states (bert base  num_hidden_states=768)
        # delta       : shape - batch_size, 1. compute_convergence_delta
        attributions, delta = lig.attribute(
            inputs=input_ids_batch,
            baselines=ref_input_ids_batch,
            target=target,
            additional_forward_args=additional_forward_args,
            # Setting this slows down the processing as it splits the example batch
            # into examples and process one at a time
            # internal_batch_size=batch_size,

            #  If the approximation error is large,
            # we can try a larger number of integral approximation steps by setting n_steps to a larger value
            n_steps=n_steps,
            return_convergence_delta=True,
        )
        # shape: batch_size, num_tokens
        # This calculates the normalized attribution for each token
        attributions_sum = summarize_attributions(attributions, batch_size)
    else:
        raise NotImplementedError

    responses = []
    for i in range(batch_size):
        response = {}
        # Remove padding
        all_tokens = get_word_token(input_ids_batch, tokenizer, i)
        words = list(
            filter(lambda word: word != tokenizer.pad_token, all_tokens))
        attributions = attributions_sum[i].tolist()[:len(words)]

        response["words"] = words
        response["importances"] = attributions
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
    input_batch = {"input_ids": input_ids_batch,
                   "attention_mask": attention_mask_batch,
                   "token_type_ids": token_type_ids_batch,
                   }
    logits_list, _ = run_inference(input_batch, mode, model)
    return torch.stack(logits_list)


def captum_sequence_forward_char_bert(
    input_ids_batch, attention_mask_batch, token_type_ids_batch,
        char_input_ids_batch, start_ids_batch, end_ids_batch,
        indexes, mode: str, model):
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
    input_batch = {"input_ids": input_ids_batch,
                   "attention_mask": attention_mask_batch,
                   "token_type_ids": token_type_ids_batch,
                   "char_input_ids": char_input_ids_batch,
                   "start_ids": start_ids_batch,
                   "end_ids": end_ids_batch,
                   }
    logits_list, _ = run_inference(input_batch, mode, model)
    return torch.stack(logits_list)


def summarize_attributions(attributions, batch_size):
    """Summarises the attribution across multiple runs
    Args:
        attributions ([list): attributions from the Layer Integrated Gradients
    Returns:
        list : Returns the attributions after normalizing them.
    """
    attributions = attributions.sum(dim=-1)
    if batch_size > 1:
        attributions = attributions.squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions
