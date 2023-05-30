from transformers import RobertaModel
from transformers import RobertaTokenizer

from torch_adapters.utils import add_prefix_tuning_embedding

# TODO move outside or in an actual test case

model = RobertaModel.from_pretrained("roberta-base")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

inputs = tokenizer.batch_encode_plus(["Hello, this is only a test", "This is also a test!"], padding=True,
                                     return_tensors="pt")

add_prefix_tuning_embedding(model, {"word_embeddings": "word",
                                    "token_type_embeddings": "token_type",
                                    "position_embeddings": "position"},
                            {"prefix_length": 30, "hidden_rank": 512})
print()
