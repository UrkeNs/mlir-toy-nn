# MLIR Toy Neural Network Compiler (Toy-NN)

## Introduction

This project extends the official **MLIR “Toy” tutorial language** into a small experimental language that supports **neural network creation, training, and inference**.

The original Toy language provided by MLIR is intentionally very simple: it supports basic tensor operations and exists mainly to demonstrate how to build a compiler using MLIR.
This project builds on that foundation and turns Toy into a **neural-network-aware language**, capable of expressing and executing a complete ML workflow:

- defining a neural network model
- training the model on a dataset

The main goal of this project is educational and experimental: to explore how MLIR dialects, types, lowering passes, and JIT execution can be used to represent and execute neural-network computations.

---

## Neural Network Overview

The neural network implemented in Toy-NN is a **deep feed-forward (fully connected) neural network**.

### Key characteristics

- **Architecture:** Sequential dense layers
- **Activation function:** `tanh`
- **Training method:** Batch Gradient Descent (BGD)
- **Loss function:** Mean Squared Error (MSE)

The model is trained end-to-end using the provided dataset tensor.
After training, the updated model can be reused for prediction without retraining.

---

## Toy-NN Language Syntax

### Example Program

```toy
def main(dataset) {
  var model = create_model(2, 4, 24, 1);
  var modelTrained = train(model, dataset, 100000, 0.1);
  var inputForPredict = ([[0.234, 0.34]]);
  var y = predict(modelTrained, inputForPredict);
  print(y);
  return;
}
```

---

### Operation Semantics

#### create_model

```toy
create_model(input, depth, width, output)
```

Creates a feed-forward neural network with:


---

#### train


```toy
train(model, dataset, epochs, learning_rate)
```

Trains the neural network and returns a **new model handle** that is used later for predict.

- `model` – model returned by `create_model`
- `dataset` – rank-2 tensor shaped as:
  `[input_dim + output_dim]`
  - first `input_dim` columns → input features
  - last  `output_dim` columns → target values
- `epochs` – number of training iterations
- `learning_rate` – gradient descent step size

---

#### predict

```toy
predict(model, input)
```

Runs a forward pass of the network.

- `model` – trained model
- `input` – rank-2 tensor `[input_dim]`

Returns a tensor containing predictions.

---

#### print

Prints tensors or memrefs depending on the lowering stage.

---

#### dataset

  - Dataset must be a .csv format
  - In this exapmle we create_model(**2**, 4, 24, **1**)
  - Every scalar is devided by comma
  - One row represents one set of inputs and expected outputs

  ```
  -1.0, -1.0, 0.761594
  -1.0, -0.8, 0.664037
  -1.0, -0.6, 0.537050 
  ```
---
## Project Status and Known Limitations

This project is a **prototype / proof-of-concept** and is not production quality.

### Known issues

- **Negative tensor literals are not supported**
  ```toy
  var predictInput = ([[-0.234, 0.34]]);
  ```
- **Mistake when modeling create_model function**
  
I initially modeled the `depth` parameter incorrectly.

In the current implementation, `depth` counts **all layers**, including:
- the **input layer**
- the **output layer**

For example:

```toy
create_model(2, 4, 24, 1)
```

is currently interpreted as:
- 1 input layer
- 2 hidden layers
- 1 output layer

However, this interpretation is **incorrect**.

#### Correct interpretation

- `depth` should count **only the layers where the activation function is applied**, i.e. the **hidden layers**
- The **input layer** should not be counted as part of the depth
- The **output layer** is typically treated separately (often without an activation function, or with a task-specific one)

With the correct definition, the example above should represent:

- **depth = 2** (two hidden layers), not 4

This will be fixed in future revisions by redefining `depth` to explicitly mean the **number of hidden layers**.

- **Suboptimal runtime behavior**
  - Some code paths are not fully optimized
  - Known memory leaks exist
  - Allocation and ownership handling can be improved

- **General fragility**
  - Limited error handling
  - Incomplete shape validation

Despite these limitations, the project successfully demonstrates a full MLIR-based neural-network compilation pipeline.

---


