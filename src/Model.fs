/// A decoder-only transformer (GPT style) built on the autograd engine.
module Model

open System
open System.IO
open Tensor

type Config =
    { Vocab: int
      BlockSize: int
      DModel: int
      Heads: int
      Layers: int }

type Layer =
    { Ln1G: Tensor; Ln1B: Tensor
      Wq: Tensor; Bq: Tensor
      Wk: Tensor; Bk: Tensor
      Wv: Tensor; Bv: Tensor
      Wo: Tensor; Bo: Tensor
      Ln2G: Tensor; Ln2B: Tensor
      W1: Tensor; B1: Tensor
      W2: Tensor; B2: Tensor }

type Gpt =
    { Config: Config
      TokEmb: Tensor
      PosEmb: Tensor
      Blocks: Layer list
      LnfG: Tensor
      LnfB: Tensor
      Head: Tensor
      HeadB: Tensor }

let private gaussian (rng: Random) (std: float32) n =
    Array.init n (fun _ ->
        let u1 = 1.0 - rng.NextDouble()
        let u2 = rng.NextDouble()
        float32 (sqrt (-2.0 * log u1) * cos (2.0 * Math.PI * u2)) * std)

let create (cfg: Config) (seed: int) =
    let rng = Random(seed)
    let d = cfg.DModel
    let w rows cols = param rows cols (gaussian rng 0.02f (rows * cols))
    let zerosP rows cols = param rows cols (Array.zeroCreate (rows * cols))
    let onesP cols = param 1 cols (Array.create cols 1.0f)
    let layer () =
        { Ln1G = onesP d; Ln1B = zerosP 1 d
          Wq = w d d; Bq = zerosP 1 d
          Wk = w d d; Bk = zerosP 1 d
          Wv = w d d; Bv = zerosP 1 d
          Wo = w d d; Bo = zerosP 1 d
          Ln2G = onesP d; Ln2B = zerosP 1 d
          W1 = w d (4 * d); B1 = zerosP 1 (4 * d)
          W2 = w (4 * d) d; B2 = zerosP 1 d }
    { Config = cfg
      TokEmb = w cfg.Vocab d
      PosEmb = w cfg.BlockSize d
      Blocks = List.init cfg.Layers (fun _ -> layer ())
      LnfG = onesP d
      LnfB = zerosP 1 d
      Head = w d cfg.Vocab
      HeadB = zerosP 1 cfg.Vocab }

let parameters (m: Gpt) =
    [ yield m.TokEmb
      yield m.PosEmb
      for l in m.Blocks do
          yield! [ l.Ln1G; l.Ln1B; l.Wq; l.Bq; l.Wk; l.Bk; l.Wv; l.Bv; l.Wo; l.Bo; l.Ln2G; l.Ln2B; l.W1; l.B1; l.W2; l.B2 ]
      yield m.LnfG
      yield m.LnfB
      yield m.Head
      yield m.HeadB ]

let parameterCount (m: Gpt) = parameters m |> List.sumBy (fun p -> p.Size)

let private linear (x: Tensor) (w: Tensor) (b: Tensor) = add (matmul x w) b

/// Multi-head causal self-attention over one sequence of T rows.
let private attention (cfg: Config) (l: Layer) (q: Tensor) (k: Tensor) (v: Tensor) =
    let dh = cfg.DModel / cfg.Heads
    let factor = 1.0f / sqrt (float32 dh)
    let heads =
        [ for h in 0 .. cfg.Heads - 1 ->
              let qh = sliceCols q (h * dh) dh
              let kh = sliceCols k (h * dh) dh
              let vh = sliceCols v (h * dh) dh
              let scores = matmul qh (transpose kh) |> scale factor |> causalMask |> softmaxRows
              matmul scores vh ]
    concatCols heads

/// Forward pass over a batch of token sequences (all of the same length T <= BlockSize); returns logits (B*T x Vocab).
let forward (m: Gpt) (batch: int[][]) =
    let cfg = m.Config
    let t = batch.[0].Length
    if t > cfg.BlockSize then failwithf "sequence length %d exceeds block size %d" t cfg.BlockSize
    let positions = sliceRows m.PosEmb 0 t
    let mutable x = concatRows [ for tokens in batch -> add (embedding m.TokEmb tokens) positions ]
    for l in m.Blocks do
        let h = layerNorm x l.Ln1G l.Ln1B
        let q = linear h l.Wq l.Bq
        let k = linear h l.Wk l.Bk
        let v = linear h l.Wv l.Bv
        let perSequence =
            [ for b in 0 .. batch.Length - 1 ->
                  attention cfg l (sliceRows q (b * t) t) (sliceRows k (b * t) t) (sliceRows v (b * t) t) ]
        x <- add x (linear (concatRows perSequence) l.Wo l.Bo)
        let h2 = layerNorm x l.Ln2G l.Ln2B
        x <- add x (linear (gelu (linear h2 l.W1 l.B1)) l.W2 l.B2)
    let final = layerNorm x m.LnfG m.LnfB
    linear final m.Head m.HeadB

/// Greedy or sampled continuation of a prompt.
let generate (m: Gpt) (prompt: int[]) (count: int) (temperature: float32) (rng: Random option) =
    let tokens = ResizeArray<int>(prompt)
    for _ in 1 .. count do
        let context = tokens.ToArray() |> Array.skip (max 0 (tokens.Count - m.Config.BlockSize))
        let logits = forward m [| context |]
        let last = sliceRows logits (context.Length - 1) 1
        let scaled = scale (1.0f / max temperature 1e-4f) last
        let probs = (softmaxRows scaled).Data
        let next =
            match rng with
            | None -> probs |> Array.indexed |> Array.maxBy snd |> fst
            | Some r ->
                let u = float32 (r.NextDouble())
                let mutable acc = 0.0f
                let mutable chosen = probs.Length - 1
                let mutable i = 0
                while i < probs.Length do
                    acc <- acc + probs.[i]
                    if acc >= u then
                        chosen <- i
                        i <- probs.Length
                    i <- i + 1
                chosen
        tokens.Add next
    tokens.ToArray() |> Array.skip prompt.Length

/// AdamW optimiser state.
type Adam =
    { M: float32[][]
      V: float32[][]
      mutable Step: int }

let adam (ps: Tensor list) =
    { M = ps |> List.map (fun p -> Array.zeroCreate p.Size) |> List.toArray
      V = ps |> List.map (fun p -> Array.zeroCreate p.Size) |> List.toArray
      Step = 0 }

/// One AdamW update; weight decay applies to matrices only (not biases or layer norm gains).
let adamStep (opt: Adam) (ps: Tensor list) (lr: float32) (weightDecay: float32) =
    opt.Step <- opt.Step + 1
    let b1, b2 = 0.9f, 0.99f
    let c1 = 1.0f - pown b1 opt.Step
    let c2 = 1.0f - pown b2 opt.Step
    ps
    |> List.iteri (fun i p ->
        let m = opt.M.[i]
        let v = opt.V.[i]
        let decay = if p.Rows > 1 then weightDecay else 0.0f
        for j in 0 .. p.Size - 1 do
            let g = p.Grad.[j]
            m.[j] <- b1 * m.[j] + (1.0f - b1) * g
            v.[j] <- b2 * v.[j] + (1.0f - b2) * g * g
            let mhat = m.[j] / c1
            let vhat = v.[j] / c2
            p.Data.[j] <- p.Data.[j] - lr * (mhat / (sqrt vhat + 1e-8f) + decay * p.Data.[j]))

/// Gradient norm clipping over all parameters.
let clipGradients (ps: Tensor list) (maxNorm: float32) =
    let total = ps |> List.sumBy (fun p -> p.Grad |> Array.sumBy (fun g -> g * g)) |> sqrt
    if total > maxNorm then
        let s = maxNorm / total
        for p in ps do
            for j in 0 .. p.Size - 1 do
                p.Grad.[j] <- p.Grad.[j] * s
    total

let save (m: Gpt) (path: string) =
    use w = new BinaryWriter(File.Create path)
    w.Write "fsgpt1"
    w.Write m.Config.Vocab
    w.Write m.Config.BlockSize
    w.Write m.Config.DModel
    w.Write m.Config.Heads
    w.Write m.Config.Layers
    for p in parameters m do
        for v in p.Data do
            w.Write v

let load (path: string) =
    use r = new BinaryReader(File.OpenRead path)
    if r.ReadString() <> "fsgpt1" then failwith "not an fsgpt model file"
    let cfg =
        { Vocab = r.ReadInt32()
          BlockSize = r.ReadInt32()
          DModel = r.ReadInt32()
          Heads = r.ReadInt32()
          Layers = r.ReadInt32() }
    let m = create cfg 0
    for p in parameters m do
        for j in 0 .. p.Size - 1 do
            p.Data.[j] <- r.ReadSingle()
    m
