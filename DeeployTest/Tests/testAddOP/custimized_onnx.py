
import torch
import torch.nn as nn
import numpy as np
import onnx

# Define constants
LEN = 10
DIM = 32


# Custom Autograd Function - This prevents ONNX from tracing through the operation
class CustomnNop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass: apply scaling and bias
        ctx is used to save tensors for backward pass
        """
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients
        """
        x = ctx.saved_tensors
        grad_x = grad_output
        return grad_x

    @staticmethod
    def symbolic(g, x):
        """
        ONNX symbolic function - defines how to export this as a black box.
        This is called during ONNX export and creates a custom operator node.
        """
        # Create a custom operator node that ONNX will NOT decompose
        # The domain "custom_domain" makes it a custom operator
        # The op_type "CustomScaleBias" is the name of your custom operation
        return g.op("custom_domain::CustomNOP",
                    x,
                    outputs=1)


# Custom layer that uses the autograd function
class CustomLayer(nn.Module):
    def __init__(self, dim):
        super(CustomLayer, self).__init__()

    def forward(self, x):
        # Use the custom autograd function
        # This ensures ONNX treats it as a black box
        return CustomnNop.apply(x)


# Define the neural network
class SimpleNetwork(nn.Module):
    def __init__(self, len_size, dim_size):
        super(SimpleNetwork, self).__init__()
        # First layer: Linear layer
        # Input: (batch, LEN, DIM) -> Output: (batch, LEN, DIM)
        self.linear = nn.Linear(dim_size, dim_size)

        # Second layer: Custom layer
        self.custom = CustomLayer(dim_size)

        # Third layer: ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, LEN, DIM)
        x = self.linear(x)
        x = self.custom(x)
        x = self.relu(x)
        return x


# Create the network
model = SimpleNetwork(LEN, DIM)
model.eval()

print("Model architecture:")
print(model)
print()

# Generate random input
# Shape: (batch=1, LEN=10, DIM=32)
input_data = torch.randn(1, LEN, DIM)

print(f"Input shape: {input_data.shape}")

# Dictionary to store tensor shapes during execution
tensor_shapes_runtime = {}

# Register forward hooks to capture intermediate tensor shapes
def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            tensor_shapes_runtime[name + '_output'] = [int(s) for s in output.shape]
        if isinstance(input, tuple) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            tensor_shapes_runtime[name + '_input'] = [int(s) for s in input[0].shape]
    return hook

# Register hooks for each layer
model.linear.register_forward_hook(make_hook('linear'))
model.custom.register_forward_hook(make_hook('custom'))
model.relu.register_forward_hook(make_hook('relu'))

# Test the network and capture shapes
with torch.no_grad():
    output_data = model(input_data)

# Record input and output shapes
tensor_shapes_runtime['model_input'] = list(input_data.shape)
tensor_shapes_runtime['model_output'] = list(output_data.shape)

print(f"Output shape: {output_data.shape}")
print(f"Input min/max: {input_data.min():.4f} / {input_data.max():.4f}")
print(f"Output min/max: {output_data.min():.4f} / {output_data.max():.4f}")
print()

# Convert to numpy for saving
input_numpy = input_data.numpy()
output_numpy = output_data.numpy()

# Save inputs and outputs to npz files
np.savez('inputs.npz', input=input_numpy)
np.savez('outputs.npz', output=output_numpy)

print("Saved inputs to inputs.npz")
print("Saved outputs to outputs.npz")
print()

# Export to ONNX with custom operator (no dynamic axes - use concrete batch size)
onnx_file = "model.onnx"
torch.onnx.export(
    model,
    input_data,
    onnx_file,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    custom_opsets={"custom_domain": 1}
)

print(f"Exported model to {onnx_file}")
print()

# Annotate ONNX model with tensor sizes captured from PyTorch execution
from onnx import helper, TensorProto

onnx_model = onnx.load(onnx_file)
print("Annotating ONNX model with tensor shapes from PyTorch execution...")

# Build a mapping from layer name to tensor name in ONNX
# We'll annotate all intermediate tensors with runtime shapes
layer_to_output_mapping = {}

for i, node in enumerate(onnx_model.graph.node):
    # Map each node to its output tensor name(s)
    if len(node.output) > 0:
        output_name = node.output[0]

        # Determine which runtime shape to use based on node type
        runtime_shape = None
        shape_with_symbolic = None

        if node.op_type == "MatMul" or node.op_type == "Add":
            # Linear layer consists of MatMul + Add
            if 'linear_output' in tensor_shapes_runtime:
                runtime_shape = tensor_shapes_runtime['linear_output']
        elif node.op_type == "CustomNOP" and node.domain == "custom_domain":
            runtime_shape = tensor_shapes_runtime.get('custom_output', None)
        elif node.op_type == "Relu":
            runtime_shape = tensor_shapes_runtime.get('relu_output', None)

        if runtime_shape:
            # Remove any existing value_info for this tensor
            to_remove = []
            for v in onnx_model.graph.value_info:
                if v.name == output_name:
                    to_remove.append(v)
            for v in to_remove:
                onnx_model.graph.value_info.remove(v)

            # Create shape with concrete values (no symbolic dimensions)
            shape_with_symbolic = []
            for i, dim_size in enumerate(runtime_shape):
                shape_with_symbolic.append(int(dim_size))

            # Add value_info with runtime shape
            output_value_info = helper.make_tensor_value_info(
                output_name,
                TensorProto.FLOAT,
                shape_with_symbolic
            )
            onnx_model.graph.value_info.append(output_value_info)

onnx_model_with_shapes = onnx_model

# Also annotate the output tensor
for output in onnx_model_with_shapes.graph.output:
    output_name = output.name
    runtime_shape = tensor_shapes_runtime.get('model_output', None)

    if runtime_shape:
        # Create shape with concrete values (no symbolic dimensions)
        shape_with_symbolic = []
        for i, dim_size in enumerate(runtime_shape):
            shape_with_symbolic.append(int(dim_size))

        # Update the output tensor type with shape
        output.type.tensor_type.Clear()
        output.type.tensor_type.elem_type = TensorProto.FLOAT
        for dim in shape_with_symbolic:
            d = output.type.tensor_type.shape.dim.add()
            if isinstance(dim, str):
                d.dim_param = dim
            else:
                d.dim_value = dim

# Save the annotated model
onnx_file_annotated = "network.onnx"
onnx.save(onnx_model_with_shapes, onnx_file_annotated)
print(f"Saved model with shape annotations to {onnx_file_annotated}")
print(f"Tensor shapes captured during execution: {tensor_shapes_runtime}")

