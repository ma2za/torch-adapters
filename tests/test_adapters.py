from transformers import RobertaForSequenceClassification, RobertaConfig
from transformers import RobertaTokenizer

from torch_adapters.adapters.prompt_tuning import prompt_attention_mask
from torch_adapters.utils import add_prompt_tuning, train_adapters, add_lora

# TODO move outside or in an actual test case

config = RobertaConfig.from_pretrained("roberta-base", num_labels=6)

model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=config)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

inputs = tokenizer.batch_encode_plus(["Hello, this is only a test", "This is also a test!"], padding=True,
                                     return_tensors="pt")

add_lora(model, ["key", "value"], {"alpha": 8, "r": 8})

add_prompt_tuning(model, {"word_embeddings": "word",
                          "token_type_embeddings": "token_type",
                          "position_embeddings": "position"},
                  {"prompt_length": 30})

train_adapters(model, ["lora", "classifier", "prompt"])

inputs["attention_mask"] = prompt_attention_mask(inputs["attention_mask"], 30)
model(**inputs)
print()
