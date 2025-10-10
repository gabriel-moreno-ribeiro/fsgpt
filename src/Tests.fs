/// Self-contained test suite: `dotnet run -- test`.
module Tests

open System
open Tensor
open Model
open Tasks

let mutable private passed = 0
let mutable private failed = 0

let check (name: string) (condition: bool) (detail: string) =
    if condition then passed <- passed + 1
    else
        failed <- failed + 1
        printfn "FAIL: %s %s" name detail

let near name (actual: float32) (expected: float32) (tol: float32) =
    check name (abs (actual - expected) <= tol) (sprintf "expected %g, got %g" expected actual)

let private rng = Random(7)
let private randomTensor rows cols requiresGrad =
    Tensor(rows, cols, Array.init (rows * cols) (fun _ -> float32 (rng.NextDouble() * 2.0 - 1.0)), requiresGrad)

/// Compares autograd gradients of `f` with central finite differences over every element of `inputs`.
let gradientCheckWith (eps: float32) (name: string) (inputs: Tensor list) (f: unit -> Tensor) =
    let loss = f ()
    backward loss
    let analytic = inputs |> List.map (fun t -> Array.copy t.Grad)
    // float32 finite differences are noisy for tiny gradients, so errors are judged against the gradient scale
    let scale = analytic |> List.map (fun g -> g |> Array.map abs |> Array.max) |> List.max
    let mutable worst = 0.0f
    inputs
    |> List.iteri (fun ti t ->
        for i in 0 .. t.Size - 1 do
            let original = t.Data.[i]
            t.Data.[i] <- original + eps
            let plus = (f ()).Data.[0]
            t.Data.[i] <- original - eps
            let minus = (f ()).Data.[0]
            t.Data.[i] <- original
            let numeric = (plus - minus) / (2.0f * eps)
            let err = abs (numeric - analytic.[ti].[i]) / max (abs numeric + abs analytic.[ti].[i]) (0.2f * scale + 1e-3f)
            worst <- max worst err)
    check (sprintf "gradient check: %s" name) (worst < 0.05f) (sprintf "worst relative error %g" worst)

let gradientCheck name inputs f = gradientCheckWith 1e-2f name inputs f

let tensorTests () =
    let a = Tensor.create 2 3 [| 1.f; 2.f; 3.f; 4.f; 5.f; 6.f |]
    let b = Tensor.create 3 2 [| 7.f; 8.f; 9.f; 10.f; 11.f; 12.f |]
    let c = matmul a b
    check "matmul values" (c.Data = [| 58.f; 64.f; 139.f; 154.f |]) (sprintf "%A" c.Data)
    let bias = Tensor.create 1 2 [| 1.f; -1.f |]
    check "broadcast add" ((add c bias).Data = [| 59.f; 63.f; 140.f; 153.f |]) ""
    check "transpose" ((transpose a).Data = [| 1.f; 4.f; 2.f; 5.f; 3.f; 6.f |]) ""
    let sm = softmaxRows (Tensor.create 1 3 [| 1.f; 2.f; 3.f |])
    near "softmax sums to one" (Array.sum sm.Data) 1.0f 1e-6f
    check "softmax ordering" (sm.Data.[2] > sm.Data.[1] && sm.Data.[1] > sm.Data.[0]) ""
    let masked = causalMask (Tensor.create 3 3 (Array.create 9 1.0f))
    check "causal mask keeps lower triangle" (masked.Data.[3] = 1.0f && masked.Data.[1] < -1e8f && masked.Data.[8] = 1.0f) ""
    let ln = layerNorm (Tensor.create 2 4 [| 1.f; 2.f; 3.f; 4.f; 10.f; 10.f; 10.f; 12.f |]) (Tensor.create 1 4 (Array.create 4 1.0f)) (Tensor.create 1 4 (Array.zeroCreate 4))
    near "layernorm row mean" (Array.average ln.Data.[0..3]) 0.0f 1e-5f
    near "layernorm row variance" (ln.Data.[0..3] |> Array.averageBy (fun v -> v * v)) 1.0f 1e-3f
    let emb = embedding (Tensor.create 3 2 [| 0.f; 1.f; 2.f; 3.f; 4.f; 5.f |]) [| 2; 0 |]
    check "embedding lookup" (emb.Data = [| 4.f; 5.f; 0.f; 1.f |]) ""
    let sc = sliceCols a 1 2
    check "sliceCols" (sc.Data = [| 2.f; 3.f; 5.f; 6.f |]) ""
    let cc = concatCols [ a; sc ]
    check "concatCols" (cc.Data = [| 1.f; 2.f; 3.f; 2.f; 3.f; 4.f; 5.f; 6.f; 5.f; 6.f |]) (sprintf "%A" cc.Data)
    let ce = crossEntropy (Tensor.create 2 3 [| 0.f; 0.f; 0.f; 5.f; 0.f; 0.f |]) [| 1; 0 |] [| 1.f; 1.f |]
    near "cross entropy of uniform and confident rows" ce.Data.[0] ((log 3.0f + 0.0134f) / 2.0f) 1e-3f

let gradientTests () =
    let x = randomTensor 3 4 true
    let w = randomTensor 4 5 true
    let b = randomTensor 1 5 true
    let g = randomTensor 1 5 true
    let beta = randomTensor 1 5 true
    let targets = [| 1; 4; 2 |]
    let weights = [| 1.f; 0.5f; 1.f |]
    gradientCheck "linear + gelu + layernorm + cross entropy" [ x; w; b; g; beta ] (fun () ->
        crossEntropy (layerNorm (gelu (add (matmul x w) b)) g beta) targets weights)
    let q = randomTensor 4 6 true
    let k = randomTensor 4 6 true
    let v = randomTensor 4 6 true
    gradientCheck "causal attention" [ q; k; v ] (fun () ->
        let heads =
            [ for h in 0 .. 1 ->
                  let qh = sliceCols q (h * 3) 3
                  let kh = sliceCols k (h * 3) 3
                  let vh = sliceCols v (h * 3) 3
                  matmul (softmaxRows (causalMask (scale 0.6f (matmul qh (transpose kh))))) vh ]
        crossEntropy (concatCols heads) [| 0; 3; 5; 1 |] [| 1.f; 1.f; 1.f; 1.f |])
    let table = randomTensor 5 3 true
    let p = randomTensor 3 4 true
    gradientCheck "embedding + mul + slices + concatRows" [ table; p ] (fun () ->
        let e = embedding table [| 4; 1; 4 |]
        let rows = concatRows [ sliceRows e 0 2; sliceRows e 1 2 ]
        let y = mul (matmul rows p) (matmul rows p)
        crossEntropy y [| 0; 1; 2; 3 |] [| 1.f; 1.f; 1.f; 1.f |])
    let m = Model.create { Vocab = 6; BlockSize = 5; DModel = 8; Heads = 2; Layers = 1 } 3
    // the default 0.02 initialisation is smaller than a finite-difference step, so scale the weights up
    for p in parameters m do
        if p.Rows > 1 then
            for i in 0 .. p.Size - 1 do
                p.Data.[i] <- p.Data.[i] * 10.0f
    gradientCheckWith 1e-3f "whole model" (parameters m) (fun () ->
        crossEntropy (forward m [| [| 0; 1; 2; 3 |]; [| 3; 2; 1; 0 |] |]) [| 1; 2; 3; 4; 2; 1; 0; 5 |] (Array.create 8 1.0f))

let modelTests () =
    let cfg = { Vocab = 7; BlockSize = 6; DModel = 16; Heads = 4; Layers = 2 }
    let m = create cfg 11
    check "parameter count" (parameterCount m = (7 * 16 + 6 * 16 + 2 * (2 * 16 + 4 * (16 * 16 + 16) + 2 * 16 + 16 * 64 + 64 + 64 * 16 + 16) + 2 * 16 + 16 * 7 + 7)) (string (parameterCount m))
    let logits1 = forward m [| [| 1; 2; 3; 4 |] |]
    let logits2 = forward m [| [| 1; 2; 3; 6 |] |]
    let same i = Array.forall2 (fun (a: float32) b -> abs (a - b) < 1e-6f) logits1.Data.[i * 7 .. i * 7 + 6] logits2.Data.[i * 7 .. i * 7 + 6]
    check "causality: earlier positions ignore later tokens" (same 0 && same 1 && same 2 && not (same 3)) ""
    let batched = forward m [| [| 1; 2; 3; 4 |]; [| 1; 2; 3; 6 |] |]
    check "batch rows equal single-sequence rows" (Array.forall2 (fun (a: float32) b -> abs (a - b) < 1e-6f) batched.Data.[.. 27] logits1.Data) ""
    let path = IO.Path.GetTempFileName()
    save m path
    let loaded = load path
    IO.File.Delete path
    check "save/load round trip" (List.forall2 (fun (a: Tensor) (b: Tensor) -> a.Data = b.Data) (parameters m) (parameters loaded)) ""
    let greedy = generate m [| 1; 2 |] 3 1.0f None
    check "generate length" (greedy.Length = 3 && Array.forall (fun t -> t >= 0 && t < 7) greedy) ""
    let sampled = generate m [| 1; 2 |] 3 0.8f (Some(Random(1)))
    check "sampled tokens in range" (Array.forall (fun t -> t >= 0 && t < 7) sampled) ""

let trainingTests () =
    let trainSet, testSet = additionDataset 1 0.0 5
    check "one-digit dataset" (trainSet.Length = 100 && testSet.Length = 0 && trainSet |> Array.forall (fun s -> s.Length = 6)) ""
    check "addition example format" (additionExample 2 7 45 = "07+45=052") (additionExample 2 7 45)
    let inputs, targets, weights = additionBatch [| "07+45=052" |]
    check "batch shapes" (inputs.[0].Length = 8 && targets.Length = 8 && weights = [| 0.f; 0.f; 0.f; 0.f; 0.f; 1.f; 1.f; 1.f |]) (sprintf "%A" weights)
    check "batch targets are shifted input" (addDecode targets = "7+45=052") (addDecode targets)
    let opts = { defaultOptions with Steps = 700; Batch = 32; Seed = 3; Every = 0 }
    let m, losses, trainAcc, _ = trainAddition 1 opts
    check "loss decreases" (Array.average losses.[.. 49] > 2.0f * Array.average losses.[losses.Length - 50 ..]) (sprintf "first %g last %g" (Array.average losses.[.. 49]) (Array.average losses.[losses.Length - 50 ..]))
    check "learns one-digit addition" (trainAcc >= 0.95) (sprintf "accuracy %g" trainAcc)
    check "solves a sum" (solveAddition m 1 3 4 = Some 7) (sprintf "%A" (solveAddition m 1 3 4))
    let data = textData "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc"
    check "text vocab" (data.Chars = [| 'a'; 'b'; 'c' |]) ""
    let tm = create (textConfig data 8) 2
    train tm { defaultOptions with Steps = 150; Batch = 16; Every = 0 } (textBatch data 8 16) |> ignore
    let sample = sampleText tm data "ab" 12 0.5f 1
    check "text model learns the cycle" (sample = "abcabcabcabcab") sample

let run () =
    let sw = Diagnostics.Stopwatch.StartNew()
    for name, suite in [ "tensor", tensorTests; "gradients", gradientTests; "model", modelTests; "training", trainingTests ] do
        try
            suite ()
        with e ->
            failed <- failed + 1
            printfn "FAIL: suite %s threw %s" name (string e)
    printfn "%d passed, %d failed (%.1fs)" passed failed sw.Elapsed.TotalSeconds
    if failed > 0 then 1 else 0
