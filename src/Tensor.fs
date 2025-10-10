/// A small reverse-mode automatic differentiation engine over 2D float32 tensors.
module Tensor

open System
open System.Collections.Generic

type Tensor(rows: int, cols: int, data: float32[], requiresGrad: bool) =
    static let mutable counter = 0
    let id =
        counter <- counter + 1
        counter
    do if data.Length <> rows * cols then failwithf "tensor %dx%d given %d values" rows cols data.Length
    member _.Rows = rows
    member _.Cols = cols
    member _.Data = data
    member val Grad: float32[] = Array.zeroCreate (rows * cols) with get, set
    member val RequiresGrad = requiresGrad with get, set
    member val Parents: Tensor list = [] with get, set
    member val BackwardFn: (unit -> unit) = ignore with get, set
    member _.Id = id
    member _.Item
        with get (r: int, c: int) = data.[r * cols + c]
    member t.Size = rows * cols
    override t.ToString() =
        let rowsText =
            [ for r in 0 .. min 5 (rows - 1) ->
                  [ for c in 0 .. min 7 (cols - 1) -> sprintf "%8.4f" data.[r * cols + c] ] |> String.concat " " ]
        sprintf "Tensor %dx%d\n%s" rows cols (String.concat "\n" rowsText)

let create rows cols (values: float32[]) = Tensor(rows, cols, values, false)
let zeros rows cols = Tensor(rows, cols, Array.zeroCreate (rows * cols), false)
let param rows cols (values: float32[]) = Tensor(rows, cols, values, true)

let private make rows cols (data: float32[]) (parents: Tensor list) (backward: Tensor -> unit) =
    let t = Tensor(rows, cols, data, parents |> List.exists (fun p -> p.RequiresGrad))
    t.Parents <- parents
    t.BackwardFn <- (fun () -> backward t)
    t

/// C = A @ B
let matmul (a: Tensor) (b: Tensor) =
    if a.Cols <> b.Rows then failwithf "matmul shape mismatch %dx%d @ %dx%d" a.Rows a.Cols b.Rows b.Cols
    let m, k, n = a.Rows, a.Cols, b.Cols
    let out = Array.zeroCreate (m * n)
    for i in 0 .. m - 1 do
        for p in 0 .. k - 1 do
            let aip = a.Data.[i * k + p]
            if aip <> 0.0f then
                let bo = p * n
                let oo = i * n
                for j in 0 .. n - 1 do
                    out.[oo + j] <- out.[oo + j] + aip * b.Data.[bo + j]
    make m n out [ a; b ] (fun t ->
        if a.RequiresGrad then
            for i in 0 .. m - 1 do
                for p in 0 .. k - 1 do
                    let mutable s = 0.0f
                    for j in 0 .. n - 1 do
                        s <- s + t.Grad.[i * n + j] * b.Data.[p * n + j]
                    a.Grad.[i * k + p] <- a.Grad.[i * k + p] + s
        if b.RequiresGrad then
            for i in 0 .. m - 1 do
                for p in 0 .. k - 1 do
                    let aip = a.Data.[i * k + p]
                    if aip <> 0.0f then
                        for j in 0 .. n - 1 do
                            b.Grad.[p * n + j] <- b.Grad.[p * n + j] + aip * t.Grad.[i * n + j])

/// A + B, where B may be a single row broadcast over the rows of A.
let add (a: Tensor) (b: Tensor) =
    let broadcast = b.Rows = 1 && a.Rows > 1
    if a.Cols <> b.Cols || (not broadcast && a.Rows <> b.Rows) then failwithf "add shape mismatch %dx%d + %dx%d" a.Rows a.Cols b.Rows b.Cols
    let cols = a.Cols
    let out = Array.init a.Size (fun i -> a.Data.[i] + (if broadcast then b.Data.[i % cols] else b.Data.[i]))
    make a.Rows a.Cols out [ a; b ] (fun t ->
        if a.RequiresGrad then
            for i in 0 .. out.Length - 1 do
                a.Grad.[i] <- a.Grad.[i] + t.Grad.[i]
        if b.RequiresGrad then
            for i in 0 .. out.Length - 1 do
                let bi = if broadcast then i % cols else i
                b.Grad.[bi] <- b.Grad.[bi] + t.Grad.[i])

/// Elementwise product.
let mul (a: Tensor) (b: Tensor) =
    if a.Rows <> b.Rows || a.Cols <> b.Cols then failwith "mul shape mismatch"
    let out = Array.init a.Size (fun i -> a.Data.[i] * b.Data.[i])
    make a.Rows a.Cols out [ a; b ] (fun t ->
        if a.RequiresGrad then
            for i in 0 .. out.Length - 1 do
                a.Grad.[i] <- a.Grad.[i] + t.Grad.[i] * b.Data.[i]
        if b.RequiresGrad then
            for i in 0 .. out.Length - 1 do
                b.Grad.[i] <- b.Grad.[i] + t.Grad.[i] * a.Data.[i])

let scale (s: float32) (a: Tensor) =
    let out = Array.map (fun v -> v * s) a.Data
    make a.Rows a.Cols out [ a ] (fun t ->
        if a.RequiresGrad then
            for i in 0 .. out.Length - 1 do
                a.Grad.[i] <- a.Grad.[i] + t.Grad.[i] * s)

/// GELU activation (tanh approximation).
let gelu (a: Tensor) =
    let c = float32 (sqrt (2.0 / Math.PI))
    let out = a.Data |> Array.map (fun x -> 0.5f * x * (1.0f + tanh (c * (x + 0.044715f * x * x * x))))
    make a.Rows a.Cols out [ a ] (fun t ->
        if a.RequiresGrad then
            for i in 0 .. out.Length - 1 do
                let x = a.Data.[i]
                let u = c * (x + 0.044715f * x * x * x)
                let th = tanh u
                let d = 0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th) * c * (1.0f + 3.0f * 0.044715f * x * x)
                a.Grad.[i] <- a.Grad.[i] + t.Grad.[i] * d)

let transpose (a: Tensor) =
    let out = Array.zeroCreate a.Size
    for r in 0 .. a.Rows - 1 do
        for c in 0 .. a.Cols - 1 do
            out.[c * a.Rows + r] <- a.Data.[r * a.Cols + c]
    make a.Cols a.Rows out [ a ] (fun t ->
        if a.RequiresGrad then
            for r in 0 .. a.Rows - 1 do
                for c in 0 .. a.Cols - 1 do
                    a.Grad.[r * a.Cols + c] <- a.Grad.[r * a.Cols + c] + t.Grad.[c * a.Rows + r])

/// Softmax over each row.
let softmaxRows (a: Tensor) =
    let out = Array.zeroCreate a.Size
    for r in 0 .. a.Rows - 1 do
        let o = r * a.Cols
        let mutable mx = Single.NegativeInfinity
        for c in 0 .. a.Cols - 1 do
            mx <- max mx a.Data.[o + c]
        let mutable sum = 0.0f
        for c in 0 .. a.Cols - 1 do
            let e = exp (a.Data.[o + c] - mx)
            out.[o + c] <- e
            sum <- sum + e
        for c in 0 .. a.Cols - 1 do
            out.[o + c] <- out.[o + c] / sum
    make a.Rows a.Cols out [ a ] (fun t ->
        if a.RequiresGrad then
            for r in 0 .. a.Rows - 1 do
                let o = r * a.Cols
                let mutable dot = 0.0f
                for c in 0 .. a.Cols - 1 do
                    dot <- dot + t.Grad.[o + c] * out.[o + c]
                for c in 0 .. a.Cols - 1 do
                    a.Grad.[o + c] <- a.Grad.[o + c] + out.[o + c] * (t.Grad.[o + c] - dot))

/// Layer normalisation over each row with learnable gain and bias (1 x cols).
let layerNorm (a: Tensor) (gamma: Tensor) (beta: Tensor) =
    let eps = 1e-5f
    let n = a.Cols
    let xhat = Array.zeroCreate a.Size
    let invStd = Array.zeroCreate a.Rows
    for r in 0 .. a.Rows - 1 do
        let o = r * n
        let mutable mean = 0.0f
        for c in 0 .. n - 1 do
            mean <- mean + a.Data.[o + c]
        mean <- mean / float32 n
        let mutable var = 0.0f
        for c in 0 .. n - 1 do
            let d = a.Data.[o + c] - mean
            var <- var + d * d
        var <- var / float32 n
        invStd.[r] <- 1.0f / sqrt (var + eps)
        for c in 0 .. n - 1 do
            xhat.[o + c] <- (a.Data.[o + c] - mean) * invStd.[r]
    let out = Array.init a.Size (fun i -> gamma.Data.[i % n] * xhat.[i] + beta.Data.[i % n])
    make a.Rows a.Cols out [ a; gamma; beta ] (fun t ->
        for r in 0 .. a.Rows - 1 do
            let o = r * n
            if gamma.RequiresGrad then
                for c in 0 .. n - 1 do
                    gamma.Grad.[c] <- gamma.Grad.[c] + t.Grad.[o + c] * xhat.[o + c]
            if beta.RequiresGrad then
                for c in 0 .. n - 1 do
                    beta.Grad.[c] <- beta.Grad.[c] + t.Grad.[o + c]
            if a.RequiresGrad then
                let mutable meanDx = 0.0f
                let mutable meanDxX = 0.0f
                for c in 0 .. n - 1 do
                    let dx = t.Grad.[o + c] * gamma.Data.[c]
                    meanDx <- meanDx + dx
                    meanDxX <- meanDxX + dx * xhat.[o + c]
                meanDx <- meanDx / float32 n
                meanDxX <- meanDxX / float32 n
                for c in 0 .. n - 1 do
                    let dx = t.Grad.[o + c] * gamma.Data.[c]
                    a.Grad.[o + c] <- a.Grad.[o + c] + invStd.[r] * (dx - meanDx - xhat.[o + c] * meanDxX))

let sliceRows (a: Tensor) (start: int) (count: int) =
    if start < 0 || start + count > a.Rows then failwith "sliceRows out of range"
    let out = Array.sub a.Data (start * a.Cols) (count * a.Cols)
    make count a.Cols out [ a ] (fun t ->
        if a.RequiresGrad then
            for i in 0 .. out.Length - 1 do
                a.Grad.[start * a.Cols + i] <- a.Grad.[start * a.Cols + i] + t.Grad.[i])

let sliceCols (a: Tensor) (start: int) (count: int) =
    if start < 0 || start + count > a.Cols then failwith "sliceCols out of range"
    let out = Array.init (a.Rows * count) (fun i -> a.Data.[(i / count) * a.Cols + start + i % count])
    make a.Rows count out [ a ] (fun t ->
        if a.RequiresGrad then
            for i in 0 .. out.Length - 1 do
                let ai = (i / count) * a.Cols + start + i % count
                a.Grad.[ai] <- a.Grad.[ai] + t.Grad.[i])

let concatRows (parts: Tensor list) =
    let cols = (List.head parts).Cols
    if parts |> List.exists (fun p -> p.Cols <> cols) then failwith "concatRows column mismatch"
    let rows = parts |> List.sumBy (fun p -> p.Rows)
    let out = Array.concat (parts |> List.map (fun p -> p.Data))
    make rows cols out parts (fun t ->
        let mutable offset = 0
        for p in parts do
            if p.RequiresGrad then
                for i in 0 .. p.Size - 1 do
                    p.Grad.[i] <- p.Grad.[i] + t.Grad.[offset + i]
            offset <- offset + p.Size)

let concatCols (parts: Tensor list) =
    let rows = (List.head parts).Rows
    if parts |> List.exists (fun p -> p.Rows <> rows) then failwith "concatCols row mismatch"
    let cols = parts |> List.sumBy (fun p -> p.Cols)
    let out = Array.zeroCreate (rows * cols)
    let mutable offset = 0
    for p in parts do
        for r in 0 .. rows - 1 do
            Array.blit p.Data (r * p.Cols) out (r * cols + offset) p.Cols
        offset <- offset + p.Cols
    make rows cols out parts (fun t ->
        let mutable offset = 0
        for p in parts do
            if p.RequiresGrad then
                for r in 0 .. rows - 1 do
                    for c in 0 .. p.Cols - 1 do
                        p.Grad.[r * p.Cols + c] <- p.Grad.[r * p.Cols + c] + t.Grad.[r * cols + offset + c]
            offset <- offset + p.Cols)

/// Rows of a table selected by index (an embedding lookup).
let embedding (table: Tensor) (indices: int[]) =
    let d = table.Cols
    let out = Array.zeroCreate (indices.Length * d)
    indices |> Array.iteri (fun i idx -> Array.blit table.Data (idx * d) out (i * d) d)
    make indices.Length d out [ table ] (fun t ->
        if table.RequiresGrad then
            indices
            |> Array.iteri (fun i idx ->
                for c in 0 .. d - 1 do
                    table.Grad.[idx * d + c] <- table.Grad.[idx * d + c] + t.Grad.[i * d + c]))

/// Adds a large negative number above the diagonal so softmax ignores future positions.
let causalMask (a: Tensor) =
    let out = Array.init a.Size (fun i -> if i % a.Cols > i / a.Cols then -1e9f else a.Data.[i])
    make a.Rows a.Cols out [ a ] (fun t ->
        if a.RequiresGrad then
            for i in 0 .. out.Length - 1 do
                if i % a.Cols <= i / a.Cols then
                    a.Grad.[i] <- a.Grad.[i] + t.Grad.[i])

/// Weighted mean cross-entropy between rows of logits and integer targets; returns a 1x1 tensor.
let crossEntropy (logits: Tensor) (targets: int[]) (weights: float32[]) =
    let n, v = logits.Rows, logits.Cols
    let probs = (softmaxRows (create n v logits.Data)).Data
    let total = Array.sum weights
    let mutable loss = 0.0f
    for i in 0 .. n - 1 do
        if weights.[i] > 0.0f then
            loss <- loss - weights.[i] * log (max probs.[i * v + targets.[i]] 1e-12f)
    make 1 1 [| loss / total |] [ logits ] (fun t ->
        if logits.RequiresGrad then
            let g = t.Grad.[0]
            for i in 0 .. n - 1 do
                if weights.[i] > 0.0f then
                    let w = g * weights.[i] / total
                    for c in 0 .. v - 1 do
                        let onehot = if c = targets.[i] then 1.0f else 0.0f
                        logits.Grad.[i * v + c] <- logits.Grad.[i * v + c] + w * (probs.[i * v + c] - onehot))

/// Runs reverse-mode differentiation from a scalar tensor; gradients of every node in the graph are reset first.
let backward (loss: Tensor) =
    let visited = HashSet<int>()
    let order = List<Tensor>()
    let rec visit (t: Tensor) =
        if visited.Add t.Id then
            for p in t.Parents do
                visit p
            order.Add t
    visit loss
    for t in order do
        Array.Clear(t.Grad, 0, t.Grad.Length)
    loss.Grad.[0] <- 1.0f
    for i in order.Count - 1 .. -1 .. 0 do
        let t = order.[i]
        if t.RequiresGrad then t.BackwardFn()
