/// Datasets, training loops and evaluation for the two built-in tasks: addition and character-level text.
module Tasks

open System
open Tensor
open Model

// ---------------------------------------------------------------- addition --

/// Character vocabulary for arithmetic: digits, '+', '='.
let addVocab = "0123456789+="
let addEncode (s: string) = s |> Seq.map (fun c -> addVocab.IndexOf c) |> Seq.toArray
let addDecode (tokens: int[]) = tokens |> Array.map (fun t -> addVocab.[t]) |> String

/// "ab+cd=efg" with zero padding; the model must predict the three answer digits.
let additionExample (digits: int) (a: int) (b: int) =
    let width = digits
    sprintf "%0*d+%0*d=%0*d" width a width b (width + 1) (a + b)

/// Every pair of numbers with the given digit count, shuffled and split into train/test.
let additionDataset (digits: int) (testFraction: float) (seed: int) =
    let limit = pown 10 digits
    let all = [| for a in 0 .. limit - 1 do for b in 0 .. limit - 1 -> additionExample digits a b |]
    let rng = Random(seed)
    let shuffled = all |> Array.sortBy (fun _ -> rng.Next())
    let nTest = int (float shuffled.Length * testFraction)
    shuffled.[nTest..], shuffled.[.. nTest - 1]

/// Inputs are all but the last character; targets are shifted by one; only answer positions are weighted.
let additionBatch (examples: string[]) =
    let t = examples.[0].Length - 1
    let inputs = examples |> Array.map (fun s -> addEncode s.[.. t - 1])
    let targets = examples |> Array.collect (fun s -> addEncode s.[1..])
    let eq = examples.[0].IndexOf '='
    let weights = examples |> Array.collect (fun _ -> Array.init t (fun i -> if i >= eq then 1.0f else 0.0f))
    inputs, targets, weights

/// Answers a problem such as "12+7" by greedy decoding.
let solveAddition (m: Gpt) (digits: int) (a: int) (b: int) =
    let prompt = sprintf "%0*d+%0*d=" digits a digits b
    let answer = generate m (addEncode prompt) (digits + 1) 1.0f None |> addDecode
    match Int32.TryParse answer with
    | true, n -> Some n
    | _ -> None

let additionAccuracy (m: Gpt) (digits: int) (examples: string[]) =
    let correct =
        examples
        |> Array.sumBy (fun s ->
            let a = int s.[.. digits - 1]
            let b = int s.[digits + 1 .. 2 * digits]
            if solveAddition m digits a b = Some(a + b) then 1 else 0)
    float correct / float examples.Length

// ---------------------------------------------------------------- generic training --

type TrainOptions =
    { Steps: int
      Batch: int
      LearningRate: float32
      WeightDecay: float32
      Warmup: int
      Seed: int
      Log: string -> unit
      Every: int }

let defaultOptions =
    { Steps = 1000
      Batch = 32
      LearningRate = 1e-3f
      WeightDecay = 0.01f
      Warmup = 50
      Seed = 1
      Log = ignore
      Every = 100 }

/// Trains with AdamW, warmup and cosine decay; `nextBatch` returns inputs, targets and loss weights.
let train (m: Gpt) (opts: TrainOptions) (nextBatch: Random -> int[][] * int[] * float32[]) =
    let ps = parameters m
    let opt = adam ps
    let rng = Random(opts.Seed)
    let losses = ResizeArray<float32>()
    let sw = Diagnostics.Stopwatch.StartNew()
    for step in 1 .. opts.Steps do
        let inputs, targets, weights = nextBatch rng
        let loss = crossEntropy (forward m inputs) targets weights
        backward loss
        clipGradients ps 1.0f |> ignore
        let progress = float32 (max 0 (step - opts.Warmup)) / float32 (max 1 (opts.Steps - opts.Warmup))
        let lr =
            if step <= opts.Warmup then opts.LearningRate * float32 step / float32 opts.Warmup
            else opts.LearningRate * (0.1f + 0.9f * 0.5f * (1.0f + cos (float32 Math.PI * progress)))
        adamStep opt ps lr opts.WeightDecay
        losses.Add loss.Data.[0]
        if opts.Every > 0 && (step % opts.Every = 0 || step = 1) then
            let recent = losses |> Seq.skip (max 0 (losses.Count - opts.Every)) |> Seq.average
            opts.Log(sprintf "step %5d  loss %.4f  lr %.2e  %.1fs" step recent lr sw.Elapsed.TotalSeconds)
    losses.ToArray()

let additionConfig (digits: int) =
    { Vocab = addVocab.Length
      BlockSize = 3 * digits + 2
      DModel = 64
      Heads = 4
      Layers = 2 }

/// Trains an addition model and returns it with train/test accuracies.
let trainAddition (digits: int) (opts: TrainOptions) =
    let trainSet, testSet = additionDataset digits 0.1 opts.Seed
    let m = create (additionConfig digits) opts.Seed
    opts.Log(sprintf "%d-digit addition: %d training and %d test problems, %d parameters" digits trainSet.Length testSet.Length (parameterCount m))
    let losses =
        train m opts (fun rng ->
            let picks = Array.init opts.Batch (fun _ -> trainSet.[rng.Next trainSet.Length])
            additionBatch picks)
    let sampleOf (xs: string[]) n = xs |> Array.truncate n
    let trainAcc = additionAccuracy m digits (sampleOf trainSet 500)
    let testAcc = if testSet.Length > 0 then additionAccuracy m digits (sampleOf testSet 500) else nan
    m, losses, trainAcc, testAcc

// ---------------------------------------------------------------- text --

type TextData =
    { Chars: char[]
      Index: Map<char, int>
      Tokens: int[] }

let textData (text: string) =
    let chars = text |> Seq.distinct |> Seq.sort |> Seq.toArray
    let index = chars |> Array.mapi (fun i c -> c, i) |> Map.ofArray
    { Chars = chars; Index = index; Tokens = text |> Seq.map (fun c -> index.[c]) |> Seq.toArray }

let textConfig (data: TextData) (blockSize: int) =
    { Vocab = data.Chars.Length
      BlockSize = blockSize
      DModel = 64
      Heads = 4
      Layers = 2 }

/// Random windows of the corpus; every position is a prediction target.
let textBatch (data: TextData) (blockSize: int) (batch: int) (rng: Random) =
    let starts = Array.init batch (fun _ -> rng.Next(data.Tokens.Length - blockSize - 1))
    let inputs = starts |> Array.map (fun s -> data.Tokens.[s .. s + blockSize - 1])
    let targets = starts |> Array.collect (fun s -> data.Tokens.[s + 1 .. s + blockSize])
    inputs, targets, Array.create (batch * blockSize) 1.0f

let sampleText (m: Gpt) (data: TextData) (prompt: string) (count: int) (temperature: float32) (seed: int) =
    let promptTokens = prompt |> Seq.map (fun c -> data.Index.[c]) |> Seq.toArray
    let out = generate m promptTokens count temperature (Some(Random(seed)))
    prompt + String(out |> Array.map (fun t -> data.Chars.[t]))
