import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel, RobertaConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.activations import ACT2FN
from typing import Optional
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertLayer, BertAttention, BertIntermediate, BertOutput, BertSelfAttention, BertSelfOutput
from docmodel.attention import FlashAttention


logger = logging.getLogger(__name__)

DOCMODEL_PRETRAINED_MODEL_ARCHIVE_MAP = {}

DOCMODEL_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class DocModelConfig(RobertaConfig):
    pretrained_config_archive_map = DOCMODEL_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "bert"

    def __init__(self, max_2d_position_embeddings=1024, add_linear=False, **kwargs):
        super().__init__(**kwargs)



class DocModelEmbeddings(nn.Module):
    def __init__(self, config):
        super(DocModelEmbeddings, self).__init__()

        self.config = config

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        config.max_2d_position_embeddings = 1024
        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.doc_linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.doc_linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.doc_linear3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.doc_linear4 = nn.Linear(config.hidden_size, config.hidden_size)

        self.relu = nn.ReLU()

    def forward(
        self,
        input_ids,
        bbox,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        h_position_embeddings = self.h_position_embeddings(
            torch.abs(bbox[:, :, 3] - bbox[:, :, 1])
        )
        w_position_embeddings = self.w_position_embeddings(
            torch.abs(bbox[:, :, 2] - bbox[:, :, 0])
        )

        temp_embeddings = self.doc_linear2(
            self.relu(
                self.doc_linear1(
                    left_position_embeddings
                    + upper_position_embeddings
                    + right_position_embeddings
                    + lower_position_embeddings
                    + h_position_embeddings
                    + w_position_embeddings
                )
            )
        )

        embeddings = (
            words_embeddings
            + position_embeddings
            + temp_embeddings
            + token_type_embeddings
        )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CustomBertModel(BertModel):

    def __init__(self, config, add_pooling_layer=True):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = CustomBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


class CustomBertEncoder(BertEncoder):
      
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([CustomBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    
class CustomBertLayer(BertLayer):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)


class CustomBertAttention(BertAttention):
    def __init__(self, config, position_embedding_type=None):
        super(BertAttention, self).__init__()
        self.self = FlashAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()




class DocModel(CustomBertModel):

    config_class = DocModelConfig
    pretrained_model_archive_map = DOCMODEL_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super(DocModel, self).__init__(config)
        self.embeddings = DocModelEmbeddings(config)
        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32
            # dtype=next(self.parameters()).dtype # this will trigger error when using high version torch
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, bbox, position_ids=position_ids, token_type_ids=token_type_ids
        )
        encoder_outputs = self.encoder(
            embedding_output, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class DocModelForTokenClassification(BertPreTrainedModel):
    config_class = DocModelConfig
    pretrained_model_archive_map = DOCMODEL_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.docmodel = DocModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.docmodel(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class MLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states



class DocModelForMLM(BertPreTrainedModel):
    config_class = DocModelConfig
    pretrained_model_archive_map = DOCMODEL_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = DocModel(config)
        self.lm_head = MLMHead(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.roberta.embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        bbox_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):

        outputs = self.roberta(
            input_ids,
            bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[
            2:
        ]  # Add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            outputs = (masked_lm_loss,) + outputs
        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)
