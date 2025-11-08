# fsgpt

> 🇺🇸 [English version below](#english)

Um modelo de linguagem estilo GPT em F#, do zero: um motor de diferenciação automática reversa sobre tensores 2D, um transformer decoder-only (embeddings de token e posição, self-attention causal multi-cabeça, blocos feed-forward com GELU, layer norm), o otimizador AdamW com warmup e decaimento por cosseno, e geração gulosa ou amostrada. Nenhuma biblioteca numérica; cada matmul e cada gradiente foi escrito na mão.

A tarefa embutida é aritmética: o modelo lê `12+07=` caractere por caractere e aprende a escrever `019`. Com 1500 passos ele acerta 100% das somas de dois dígitos que nunca viu. Tem também um modo que treina em qualquer arquivo de texto e gera continuações.

```sh
dotnet run -c Release -- test                      # testes: gradientes numéricos, causalidade, aprendizado
dotnet run -c Release -- train-add --steps 1500    # aprende a somar; reporta acurácia em problemas novos
dotnet run -c Release -- add 37+58                 # responde com o modelo salvo
dotnet run -c Release -- text README.md --steps 1000 --prompt "The model"
```

## Como é montado

- `src/Tensor.fs`: um tensor carrega dados, gradiente, pais e uma closure de backward. Operações: matmul, soma com broadcast, produto elemento a elemento, escala, GELU, transposição, softmax por linha, layer norm, fatias e concatenações de linhas/colunas, lookup de embedding, máscara causal e cross-entropy ponderada. `backward` ordena o grafo topologicamente e roda as closures ao contrário.
- `src/Model.fs`: o transformer. As sequências de um batch são empilhadas numa matriz pras camadas lineares e separadas por sequência pra atenção, que fatia Q/K/V em cabeças, aplica `softmax(mask(QKᵀ/√d))V` e concatena. AdamW aplica weight decay só em matrizes; gradientes são cortados por norma. Modelos salvam num binário pequeno.
- `src/Tasks.fs`: dados e loops de treino. Os problemas de adição são todos os pares de números de n dígitos, embaralhados e divididos 90/10; a loss só conta os dígitos da resposta.

Modelo padrão: 2 camadas, 64 dimensões, 4 cabeças (uns 105 mil parâmetros).

A parte que eu subestimei foi o gradient checking: comparar cada gradiente com diferenças finitas em float32 é traiçoeiro, porque pesos de 0.02 são menores que um passo de 0.01. O teste do modelo inteiro escala os pesos e usa um passo menor, e aí sim bate.

Testes: `dotnet run -- test` (operações contra valores conhecidos, todo gradiente comparado com diferenças finitas centrais numa cadeia linear/GELU/layer norm, num bloco de atenção causal de duas cabeças, em embeddings com fatias e no modelo inteiro; causalidade, consistência do batch, salvar/carregar, geração, formatação do dataset, a loss caindo, o modelo aprendendo adição de um dígito a 95%+, e um modelo de texto aprendendo um padrão repetido).

---

## English

A GPT-style language model in F#, from scratch: a reverse-mode automatic differentiation engine over 2D tensors, a decoder-only transformer (token and position embeddings, multi-head causal self-attention, feed-forward blocks with GELU, layer norm), the AdamW optimizer with warmup and cosine decay, and greedy or sampled generation. No numeric library; every matmul and every gradient was written by hand.

The built-in task is arithmetic: the model reads `12+07=` character by character and learns to write `019`. With 1500 steps it gets 100% of the two-digit sums it has never seen. There's also a mode that trains on any text file and generates continuations.

```sh
dotnet run -c Release -- test                      # tests: numerical gradients, causality, learning
dotnet run -c Release -- train-add --steps 1500    # learns to add; reports accuracy on new problems
dotnet run -c Release -- add 37+58                 # answers with the saved model
dotnet run -c Release -- text README.md --steps 1000 --prompt "The model"
```

## How it's put together

- `src/Tensor.fs`: a tensor carries data, gradient, parents and a backward closure. Operations: matmul, broadcast add, element-wise product, scaling, GELU, transpose, row-wise softmax, layer norm, row/column slices and concatenations, embedding lookup, causal mask and weighted cross-entropy. `backward` sorts the graph topologically and runs the closures in reverse.
- `src/Model.fs`: the transformer. The sequences of a batch are stacked into one matrix for the linear layers and split per sequence for attention, which slices Q/K/V into heads, applies `softmax(mask(QKᵀ/√d))V` and concatenates. AdamW applies weight decay only to matrices; gradients are clipped by norm. Models save to a small binary.
- `src/Tasks.fs`: data and training loops. The addition problems are every pair of n-digit numbers, shuffled and split 90/10; the loss only counts the digits of the answer.

Default model: 2 layers, 64 dimensions, 4 heads (about 105 thousand parameters).

The part I underestimated was gradient checking: comparing every gradient with finite differences in float32 is treacherous, because 0.02 weights are smaller than a 0.01 step. The whole-model test scales the weights and uses a smaller step, and then it matches.

Tests: `dotnet run -- test` (operations against known values, every gradient compared with central finite differences on a linear/GELU/layer norm chain, on a two-head causal attention block, on embeddings with slices and on the whole model; causality, batch consistency, save/load, generation, dataset formatting, the loss going down, the model learning one-digit addition to 95%+, and a text model learning a repeated pattern).

MIT.
