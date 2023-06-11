from transformers import (
    RobertaForSequenceClassification,
    RobertaConfig,
    RobertaTokenizer,
)

from torch_adapters.utils import add_prefix_tuning, drop_prefix_tuning_reparametrization

config = RobertaConfig.from_pretrained("roberta-base", num_labels=6)

model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=config)

add_prefix_tuning(model, ["self"], {"prefix_size": 64})

print()
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

inputs = tokenizer.batch_encode_plus(
    [
        "Hello, this is only a test",
        "This is also a test!",
        "Well maybe this is also a test!",
    ],
    padding=True,
    return_tensors="pt",
)

# add_lora(model, ["key", "value"], {"alpha": 8, "r": 8})
#
# add_adapter(model, ["output.dense"], {"adapter_size": 64})
#
# merge_lora(model, ["key", "value"])
#
# add_prompt_tuning(
#     model,
#     {
#         "word_embeddings": "word",
#         "token_type_embeddings": "token_type",
#         "position_embeddings": "position",
#     },
#     {"prompt_length": 30},
# )
#
# train_adapters(model, ["lora", "classifier", "prompt"])
#
# inputs["attention_mask"] = prompt_attention_mask(inputs["attention_mask"], 30)
#
# # Trainer()
# optimizer_grouped_parameters = [
#     {
#         "params": [
#             p
#             for n, p in model.named_parameters()
#             if (p.requires_grad and "embeddings" in n)
#         ],
#     },
#     {
#         "params": [
#             p
#             for n, p in model.named_parameters()
#             if (not p.requires_grad and "embeddings" in n)
#         ],
#         "lr": 0.0,
#     },
# ]
# model.roberta.embeddings.requires_grad_(True)
# from torch.optim import AdamW
#
# optimizer = AdamW(
#     optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-8, lr=5e-5, weight_decay=0
# )
#
#
drop_prefix_tuning_reparametrization(model)
model(**inputs)
print()
