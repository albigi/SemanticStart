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

// Ctrl+C has to reach the model download and the microphone, not just kill the process: a probe
// that is aborted mid-download leaves a half-written temp file that the next run has to clean up,
// and SpeechModelBootstrapper only does that cleanup on a cancelled token.
using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, eventArgs) =>
{
    // Only the first Ctrl+C is intercepted. The prompt below sits in a blocking Console.ReadLine
    // that a token cannot interrupt, so a second Ctrl+C has to be left alone to end the process -
    // otherwise asking politely would make the probe impossible to quit.
    if (cancellation.IsCancellationRequested)
        return;

    eventArgs.Cancel = true;
    cancellation.Cancel();
};

// The pipeline reports what it handled internally rather than throwing it - dropped audio buffers,
// a microphone that failed to stop. A latency probe that does not show those is reporting numbers
// measured on audio the transcriber never received, so they go to stderr, out of the way of the
// table on stdout.
SpeechDiagnostics.Reported += report => Console.Error.WriteLine(
    report.Exception is { } ex
        ? $"[{report.Operation}] {report.Message} {ex}"
        : $"[{report.Operation}] {report.Message}");

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

// A wall-clock limit on a 75 MB download would fail an honest slow connection, so what is bounded
// is silence: the deadline is pushed out on every progress report, and only a transfer that has
// actually stalled trips it. HttpClient's own 10-minute timeout covers a connection that never
// opens; this covers one that opens and then stops.
using var stalled = new CancellationTokenSource(ProbeTimeouts.DownloadStall);
using var loadCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellation.Token, stalled.Token);

var loadStopwatch = Stopwatch.StartNew();
ISpeechTranscriber transcriber;
try
{
    transcriber = await provider.CreateAsync(
        new Progress<double>(p =>
        {
            stalled.CancelAfter(ProbeTimeouts.DownloadStall);
            Console.Write($"\rDownload/load progress: {p:P0}   ");
        }),
        loadCancellation.Token);
}
catch (OperationCanceledException)
{
    Console.Error.WriteLine();
    Console.Error.WriteLine(stalled.IsCancellationRequested && !cancellation.IsCancellationRequested
        ? $"The model download made no progress for {ProbeTimeouts.DownloadStall.TotalMinutes:N0} minutes and was abandoned."
        : "Cancelled.");
    return 3;
}
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
        cancellation.Token.ThrowIfCancellationRequested();

        Console.WriteLine($"--- Utterance {i} of {iterations} ---");
        Console.Write("Press Enter, then speak a short phrase: ");
        Console.ReadLine();

        cancellation.Token.ThrowIfCancellationRequested();

        var result = await RunOneUtteranceAsync(engine, i, cancellation.Token);
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
catch (OperationCanceledException)
{
    Console.WriteLine();
    Console.WriteLine("Cancelled; reporting the utterances completed so far.");
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

static async Task<UtteranceResult> RunOneUtteranceAsync(
    DictationEngine engine,
    int index,
    CancellationToken cancellationToken)
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

    // ConfigureAwait is deliberately absent throughout: a console app has no SynchronizationContext
    // to capture, so it would be a no-op here. It is used in SemanticStart.Core, which is consumed
    // by the WPF app, where the continuation would otherwise be posted to the UI thread.
    await engine.ListenAsync(new DictationOptions(), OnTranscript, OnLevel, cancellationToken);

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

// Top-level statements: these are local functions of the generated entry point, which cannot carry
// an accessibility modifier. They are private to Program either way, and this matches
// TensorPrimitivesProbe.
static int ParseIterationCount(string[] args)
{
    const int defaultIterations = 5;
    if (args.Length == 0)
        return defaultIterations;

    if (!int.TryParse(args[0], out var parsed) || parsed <= 0)
        throw new ArgumentException($"Expected a positive utterance count, got \"{args[0]}\".");

    return parsed;
}

internal static class ProbeTimeouts
{
    /// <summary>How long the download may make no progress at all before it is abandoned.</summary>
    internal static readonly TimeSpan DownloadStall = TimeSpan.FromMinutes(2);
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
