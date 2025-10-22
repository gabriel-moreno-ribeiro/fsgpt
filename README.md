# fsgpt

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

**EN:** a GPT-style transformer in F# with a hand-written reverse-mode autograd engine over 2D tensors, multi-head causal attention, GELU MLP blocks, layer norm, AdamW with warmup and cosine decay, and greedy/sampled generation. Learns two-digit addition to 100% on unseen problems; the tests include finite-difference gradient checks of every operation and of the whole model. MIT.
