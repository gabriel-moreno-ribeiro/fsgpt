module Program

open System
open System.IO
open Model
open Tasks

let usage () =
    eprintfn "usage:"
    eprintfn "  fsgpt train-add [--digits 2] [--steps 3000] [--batch 64] [--out model.bin]   train a transformer to add numbers"
    eprintfn "  fsgpt add <a+b> [--model model.bin]                                          solve a sum with a saved model"
    eprintfn "  fsgpt text <corpus.txt> [--steps 1000] [--block 32] [--prompt text]           character-level language model"
    eprintfn "  fsgpt test                                                                    run the self-tests"

let option (args: string list) (name: string) (fallback: string) =
    let rec find =
        function
        | n :: v :: _ when n = name -> v
        | _ :: rest -> find rest
        | [] -> fallback
    find args

[<EntryPoint>]
let main argv =
    match List.ofArray argv with
    | "test" :: _ -> Tests.run ()
    | "train-add" :: rest ->
        let digits = int (option rest "--digits" "2")
        let steps = int (option rest "--steps" "3000")
        let batch = int (option rest "--batch" "64")
        let out = option rest "--out" "model.bin"
        let opts = { defaultOptions with Steps = steps; Batch = batch; Log = printfn "%s"; Every = max 1 (steps / 20) }
        let m, _, trainAcc, testAcc = trainAddition digits opts
        printfn "train accuracy %.3f  test accuracy %.3f" trainAcc testAcc
        save m out
        printfn "saved %s" out
        for a, b in [ 12, 7; 58, 46; 99, 99; 0, 5 ] do
            if a < pown 10 digits && b < pown 10 digits then
                printfn "%d + %d = %A" a b (solveAddition m digits a b)
        0
    | "add" :: problem :: rest ->
        let path = option rest "--model" "model.bin"
        if not (File.Exists path) then
            eprintfn "no model at %s; run train-add first" path
            1
        else
            let m = load path
            let digits = (m.Config.BlockSize - 2) / 3
            match problem.Split '+' with
            | [| a; b |] ->
                match solveAddition m digits (int a) (int b) with
                | Some n -> printfn "%s = %d" problem n
                | None -> printfn "%s = ? (model produced a non-number)" problem
                0
            | _ ->
                eprintfn "expected a problem like 12+7"
                2
    | "text" :: path :: rest ->
        let steps = int (option rest "--steps" "1000")
        let block = int (option rest "--block" "32")
        let prompt = option rest "--prompt" ""
        let text = File.ReadAllText path
        let data = textData text
        printfn "corpus: %d characters, %d distinct" text.Length data.Chars.Length
        let m = create (textConfig data block) 1
        train m { defaultOptions with Steps = steps; Batch = 32; Log = printfn "%s"; Every = max 1 (steps / 10) } (textBatch data block 32) |> ignore
        let start = if prompt = "" then text.Substring(0, min 8 text.Length) else prompt
        for temperature in [ 0.5f; 0.8f ] do
            printfn "\n--- temperature %.1f ---\n%s" temperature (sampleText m data start 300 temperature 1)
        0
    | _ ->
        usage ()
        2
