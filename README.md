# fsgpt

A GPT-style language model written from scratch in F#: a reverse-mode
automatic differentiation engine over 2D tensors, a decoder-only
transformer (token and position embeddings, multi-head causal
self-attention, GELU feed-forward blocks, layer normalisation), the AdamW
optimiser with warmup and cosine decay, and greedy or sampled generation.
No numerical libraries; every matmul and every gradient is hand-written.

The built-in task is arithmetic: the model reads `12+07=` one character
at a time and learns to write `019`. A second mode trains a
character-level model on any text file and samples from it.

```sh
dotnet run -c Release -- test                      # self-tests: numeric gradient checks, causality, learning
dotnet run -c Release -- train-add --steps 3000    # learns two-digit addition, reports held-out accuracy
dotnet run -c Release -- add 37+58                 # answers with the saved model
dotnet run -c Release -- text README.md --steps 1000 --prompt "The model"
```

## Design

- `src/Tensor.fs` — tensors carry data, gradient, parents and a backward
  closure. Operations: matmul, broadcast add, elementwise mul, scale,
  GELU, transpose, row softmax, layer norm, row/column slices and
  concatenations, embedding lookup, causal mask and weighted
  cross-entropy. `backward` topologically sorts the graph and runs the
  closures in reverse.
- `src/Model.fs` — the transformer. Sequences in a batch are stacked into
  one matrix for the linear layers and split per sequence for attention,
  which slices the query/key/value matrices into heads, applies
  `softmax(mask(QKᵀ/√d))V` and concatenates. AdamW applies weight decay to
  matrices only; gradients are norm-clipped. Models save to a small binary
  file.
- `src/Tasks.fs` — datasets and training loops. Addition problems are
  every pair of n-digit numbers, shuffled and split 90/10; the loss is
  weighted so only the answer digits count. Text mode samples random
  windows from a corpus.

Default model: 2 layers, 64-dimensional, 4 heads (about 105k parameters
for the addition vocabulary). Two-digit addition reaches over 90% exact
answers on unseen problems after a few thousand steps on a CPU.

## Tests

`dotnet run -- test` checks tensor operations against known values,
compares every autograd gradient with central finite differences
(a linear/GELU/layer-norm chain, a two-head causal attention block,
embeddings with slices and concatenations, and the whole model), verifies
causality (a later token cannot change earlier logits), batching
consistency, save/load, generation, dataset formatting, that the loss
falls, that the model learns one-digit addition to at least 95%, and that
a text model learns a repeating pattern.

## License

MIT
