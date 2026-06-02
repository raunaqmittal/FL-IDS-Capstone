import torch
from src.components.model.model import (
    MLPClassifier,
    get_model_parameters,
    set_model_parameters,
)

model = MLPClassifier(
    input_dim=78,
    hidden_dims=[64, 32, 16],
    num_classes=2,
)

x = torch.randn(8, 78)
y = torch.randint(0, 2, (8,))

logits = model(x)
print("Logits shape:", logits.shape)

loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, y)
print("Loss:", loss.item())

loss.backward()
print("Backward pass successful")

params = get_model_parameters(model)
print("Number of parameter arrays:", len(params))

new_model = MLPClassifier(
    input_dim=78,
    hidden_dims=[64, 32, 16],
    num_classes=2,
)

set_model_parameters(new_model, params)
print("Flower parameter serialization successful")
 
