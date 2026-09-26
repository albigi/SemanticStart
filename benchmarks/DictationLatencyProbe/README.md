# Dictation latency probe

Measures the two numbers that decide whether dictation feels instant or laggy: time from
microphone-open to the first partial transcript, and time to the final transcript, using the real
`WasapiMicrophoneCapture` -> `SileroVoiceActivityDetector` -> `SherpaOnnxSpeechTranscriber` ->
`DictationEngine` pipeline the app ships. There is no recorded-audio fixture here on purpose: a
fixture would measure the transcriber's compute time but not the WASAPI buffering and VAD framing
that are called out as being on the latency path in `WasapiMicrophoneCapture`'s design notes.

## Requirements

- Real Windows hardware with a working microphone. This cannot be run in CI or under any other
  headless/virtualized environment - there is no fake microphone path, and WASAPI capture has no
  meaning without an audio device.
- Network access on first run only, to download the speech model.

## Running

From the repository root, in Release:

```powershell
# x64
dotnet run -c Release -r win-x64 --project .\benchmarks\DictationLatencyProbe\DictationLatencyProbe.csproj -- 5

# ARM64
dotnet run -c Release -r win-arm64 --project .\benchmarks\DictationLatencyProbe\DictationLatencyProbe.csproj -- 5
```

The trailing `5` is the number of utterances to record; it defaults to 5 if omitted. Run the same
count on both architectures if you are comparing them, so the min/median/p95/max rows line up.

On first run the probe downloads the sherpa-onnx streaming Zipformer and Silero VAD models used by
`SpeechModelBootstrapper` - about 73 MB in total - into the app's models directory. This is a
one-time cost per machine; later runs skip straight to loading. Model load (including any
download) is timed separately from the per-utterance numbers, so a slow first download does not
pollute the latency measurements.

## What it prints

- Runtime, OS, and `RuntimeInformation.ProcessArchitecture`, so an x64 run and an ARM64 run are
  self-labelling when compared side by side.
- Model load time (download + ONNX session creation), reported once, not per utterance.
- Per utterance: time from capture start to the first partial transcript and to the final
  transcript, the same measured from the first voiced frame (an RMS-level proxy for speech onset;
  see the comment in `Program.cs` for why it is a proxy rather than the VAD's own decision), and
  the transcript text itself, so a suspicious latency number can be checked against what was
  actually recognized.
- A summary table of min/median/p95/max across all recorded utterances for both "from capture
  start" and "from first voiced frame", for both first-partial and final.

## Failure modes

If the default microphone is missing or Windows has blocked desktop-app microphone access, the
probe fails loudly with the same `MicrophoneUnavailableException` message the app itself would
show, which names the exact setting to change: **Settings > Privacy & security > Microphone >
"Let desktop apps access your microphone"**. It does not retry or fall back - a benchmark that
silently produced zero utterances would be worse than one that stops and says why.
