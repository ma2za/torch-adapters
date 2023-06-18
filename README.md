# Torch Adapters

# Introduction

Small Library of Torch Adaptation modules

### Supported Methods

- [X] LoRA
- [X] Prompt Tuning
- [X] Bottleneck Adapter
- [X] Prefix Tuning
- [ ] P-Tuning

# Installation

You can install torch-adapters using:

    $ pip install torch-adapters

# Usage

```python
from torch_adapters.utils import add_lora

# Add lora to the model
add_lora(model, ["key", "value"], {"alpha": 8, "r": 8})
```

