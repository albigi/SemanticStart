using System.Diagnostics;
using System.Runtime.InteropServices;
using SemanticStart.Core.Speech;

// Runs on real hardware only: it opens the default microphone through the same
// WasapiMicrophoneCapture/SileroVoiceActivityDetector/DictationEngine pipeline the app uses, so
// the numbers it prints are the numbers a user would actually experience, not a synthetic
// approximation of them. There is no recorded-audio mode, on purpose - a fixture would exercise
// the transcriber's compute time but not the WASAPI buffering and VAD framing that the design
// notes on WasapiMicrophoneCapture call out as being on the latency path.

var iterations = ParseIterationCount(args);

Console.WriteLine($"Runtime: {RuntimeInformation.FrameworkDescription}");
Console.WriteLine($"OS: {RuntimeInformation.OSDescription}");
Console.WriteLine($"Architecture: {RuntimeInformation.ProcessArchitecture}");
Console.WriteLine($"Utterances to record: {iterations:N0}");
Console.WriteLine();

var bootstrapper = new SpeechModelBootstrapper();
var provider = new SherpaOnnxSpeechProvider(bootstrapper, allowDownload: true);

// Cheap by contract (see ISpeechTranscriberProvider.CheckAvailability): this is the same call the
// app makes at startup, so a probe that reports "available" here is reporting what the app would
// have reported too.
var availability = provider.CheckAvailability();
if (!availability.IsAvailable)
{
    Console.Error.WriteLine($"Speech engine unavailable: {availability.Reason}");
    return 1;
}

var alreadyDownloaded = bootstrapper.IsDownloaded;
Console.WriteLine(alreadyDownloaded
    ? "Model files already present on disk."
    : $"Downloading the speech model (~{SpeechModelBootstrapper.ApproximateDownloadBytes / (1024 * 1024):N0} MB) - this only happens once.");

var loadStopwatch = Stopwatch.StartNew();
var transcriber = await provider.CreateAsync(
    new Progress<double>(p => Console.Write($"\rDownload/load progress: {p:P0}   ")),
    CancellationToken.None).ConfigureAwait(false);
loadStopwatch.Stop();
if (!alreadyDownloaded)
    Console.WriteLine();

Console.WriteLine($"Model ready in {loadStopwatch.Elapsed.TotalSeconds:F1} s (download + ONNX session creation).");
Console.WriteLine();

// DictationEngine takes ownership of both the transcriber and the VAD (see its Dispose), so they
// are not wrapped in their own `using` here - that would dispose each twice.
var voiceActivity = new SileroVoiceActivityDetector(bootstrapper.VadPath);
using var engine = new DictationEngine(transcriber, voiceActivity, () => new WasapiMicrophoneCapture());

var results = new List<UtteranceResult>(iterations);

try
{
    for (var i = 1; i <= iterations; i++)
    {
        Console.WriteLine($"--- Utterance {i} of {iterations} ---");
        Console.Write("Press Enter, then speak a short phrase: ");
        Console.ReadLine();

        var result = await RunOneUtteranceAsync(engine, i).ConfigureAwait(false);
        results.Add(result);

        Console.WriteLine($"  First partial: {Format(result.CaptureToFirstPartial)} from capture start" +
            (result.VoicedToFirstPartial is { } voicedPartial ? $", {Format(voicedPartial)} from first voiced frame" : string.Empty) +
            $" - \"{result.PartialText}\"");
        Console.WriteLine($"  Final:         {Format(result.CaptureToFinal)} from capture start" +
            (result.VoicedToFinal is { } voicedFinal ? $", {Format(voicedFinal)} from first voiced frame" : string.Empty) +
            $" - \"{result.FinalText}\"");
        Console.WriteLine();
    }
}
catch (MicrophoneUnavailableException ex)
{
    // The message already names the exact Windows setting (see
    // MicrophoneUnavailableException.DescribeFailure); repeating that string here would just be a
    // second place for it to go stale if the setting is ever renamed.
    Console.Error.WriteLine($"Microphone unavailable ({ex.Failure}): {ex.Message}");
    return 2;
}

PrintSummary(results);
return 0;

static async Task<UtteranceResult> RunOneUtteranceAsync(DictationEngine engine, int index)
{
    var stopwatch = Stopwatch.StartNew();
    TimeSpan? firstVoicedFrame = null;
    TimeSpan? firstPartial = null;
    TimeSpan? final = null;
    var partialText = string.Empty;
    var finalText = string.Empty;

    // The engine reports level (RMS) rather than the VAD's own speech decision, so "first voiced
    // frame" here is an energy-based proxy for the moment speech starts, not the exact frame the
    // endpoint detector latched onto internally. It is good enough to separate "time the user
    // spent taking a breath before speaking" from "time the engine spent producing text" without
    // reaching into DictationEngine's private endpointing state.
    const float voicedLevelThreshold = 0.05f;

    void OnLevel(float level)
    {
        if (firstVoicedFrame is null && level >= voicedLevelThreshold)
            firstVoicedFrame = stopwatch.Elapsed;
    }

    void OnTranscript(SpeechTranscript transcript)
    {
        if (transcript is PartialTranscript partial)
        {
            firstPartial ??= stopwatch.Elapsed;
            partialText = partial.Text;
        }
        else if (transcript is FinalTranscript finalTranscript)
        {
            final = stopwatch.Elapsed;
            finalText = finalTranscript.Text;
        }
    }

    await engine.ListenAsync(new DictationOptions(), OnTranscript, OnLevel).ConfigureAwait(false);

    return new UtteranceResult(
        index,
        firstPartial ?? TimeSpan.Zero,
        final ?? TimeSpan.Zero,
        firstVoicedFrame is { } voiced && firstPartial is { } partialAt ? partialAt - voiced : null,
        firstVoicedFrame is { } voiced2 && final is { } finalAt ? finalAt - voiced2 : null,
        partialText,
        finalText);
}

static void PrintSummary(IReadOnlyList<UtteranceResult> results)
{
    if (results.Count == 0)
    {
        Console.WriteLine("No utterances were recorded.");
        return;
    }

    Console.WriteLine($"=== Summary over {results.Count} utterance(s), architecture {RuntimeInformation.ProcessArchitecture} ===");
    Console.WriteLine();
    Console.WriteLine("Metric                                  Min       Median    P95       Max");
    PrintRow("Capture start -> first partial", results.Select(r => (TimeSpan?)r.CaptureToFirstPartial));
    PrintRow("Capture start -> final", results.Select(r => (TimeSpan?)r.CaptureToFinal));
    PrintRow("First voiced frame -> first partial", results.Select(r => r.VoicedToFirstPartial));
    PrintRow("First voiced frame -> final", results.Select(r => r.VoicedToFinal));
}

static void PrintRow(string label, IEnumerable<TimeSpan?> values)
{
    var samples = values.Where(v => v.HasValue).Select(v => v!.Value).OrderBy(v => v).ToArray();
    if (samples.Length == 0)
    {
        Console.WriteLine($"{label,-40} (no samples)");
        return;
    }

    var min = samples[0];
    var max = samples[^1];
    var median = Percentile(samples, 0.50);
    var p95 = Percentile(samples, 0.95);
    Console.WriteLine($"{label,-40} {Format(min),-9} {Format(median),-9} {Format(p95),-9} {Format(max)}");
}

// Nearest-rank percentile: simple, deterministic, and good enough for the small sample counts a
// manual microphone benchmark can realistically collect in one sitting.
static TimeSpan Percentile(IReadOnlyList<TimeSpan> sortedSamples, double percentile)
{
    var rank = (int)Math.Ceiling(percentile * sortedSamples.Count) - 1;
    return sortedSamples[Math.Clamp(rank, 0, sortedSamples.Count - 1)];
}

static string Format(TimeSpan value) => $"{value.TotalMilliseconds:F0} ms";

static int ParseIterationCount(string[] args)
{
    const int defaultIterations = 5;
    if (args.Length == 0)
        return defaultIterations;

    if (!int.TryParse(args[0], out var parsed) || parsed <= 0)
        throw new ArgumentException($"Expected a positive utterance count, got \"{args[0]}\".");

    return parsed;
}

/// <summary>One utterance's timing, from capture start and from the first voiced frame proxy.</summary>
internal sealed record UtteranceResult(
    int Index,
    TimeSpan CaptureToFirstPartial,
    TimeSpan CaptureToFinal,
    TimeSpan? VoicedToFirstPartial,
    TimeSpan? VoicedToFinal,
    string PartialText,
    string FinalText);
