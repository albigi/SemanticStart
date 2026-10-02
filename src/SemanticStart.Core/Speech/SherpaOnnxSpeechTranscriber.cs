using System.Runtime.CompilerServices;
using SherpaOnnx;

namespace SemanticStart.Core.Speech;

/// <summary>
/// The streaming Zipformer transducer from sherpa-onnx (Apache-2.0), running on the CPU.
///
/// <para>
/// Streaming is the whole reason for this engine rather than a batch one. Results arrive while
/// the user is still speaking, so the search box fills in as the sentence is said and the only
/// wait the user experiences is the trailing-silence window - not the length of what they said.
/// </para>
/// <para>
/// Decoding is synchronous, CPU-bound work performed on whichever thread enumerates the result
/// stream. That thread must never be the UI thread; <c>DictationSession</c> is what guarantees it.
/// </para>
/// </summary>
public sealed class SherpaOnnxSpeechTranscriber : ISpeechTranscriber
{
    private readonly OnlineRecognizer _recognizer;
    private bool _disposed;

    public SherpaOnnxSpeechTranscriber(SpeechModelFiles files, SpeechProviderMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(files);
        Metadata = metadata;
        _recognizer = new OnlineRecognizer(CreateConfig(files));
    }

    internal static OnlineRecognizerConfig CreateConfig(SpeechModelFiles files)
    {
        var config = new OnlineRecognizerConfig();
        config.FeatConfig.SampleRate = MonoFloatResampler.TargetSampleRate;
        config.FeatConfig.FeatureDim = 80;
        config.ModelConfig.Transducer.Encoder = files.EncoderPath;
        config.ModelConfig.Transducer.Decoder = files.DecoderPath;
        config.ModelConfig.Transducer.Joiner = files.JoinerPath;
        config.ModelConfig.Tokens = files.TokensPath;
        config.ModelConfig.Provider = "cpu";

        // Two threads: enough to keep decoding ahead of real time on a laptop core, few enough
        // that a dictation session does not make the rest of the machine stutter. This runs on a
        // user's foreground machine while they are waiting to search, not on a transcription box.
        config.ModelConfig.NumThreads = 2;
        // Keep alternative word sequences instead of committing to each locally best token.
        config.DecodingMethod = "modified_beam_search";
        config.MaxActivePaths = 4;

        // The recognizer's own endpointing is off: Silero decides when the utterance has ended,
        // because that decision is shared with the level meter and the listening indicator and is
        // tuned by a setting the user can see. Two endpoint rules running at once would race.
        config.EnableEndpoint = 0;

        return config;
    }

    public SpeechProviderMetadata Metadata { get; }

    public async IAsyncEnumerable<SpeechTranscript> TranscribeAsync(
        IAsyncEnumerable<ReadOnlyMemory<float>> audio,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(audio);

        using var stream = _recognizer.CreateStream();
        var lastText = string.Empty;

        await foreach (var buffer in audio.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            if (buffer.Length == 0)
                continue;

            stream.AcceptWaveform(MonoFloatResampler.TargetSampleRate, buffer.ToArray());
            while (_recognizer.IsReady(stream))
                _recognizer.Decode(stream);

            var text = SpeechTextFormatter.Format(_recognizer.GetResult(stream).Text);
            if (text.Length == 0 || string.Equals(text, lastText, StringComparison.Ordinal))
                continue;

            lastText = text;
            yield return new PartialTranscript(text);
        }

        // Supply the encoder's right context before flushing, as in sherpa's streaming examples.
        // This is synthetic silence, not an extra wait with the microphone open.
        cancellationToken.ThrowIfCancellationRequested();
        stream.AcceptWaveform(MonoFloatResampler.TargetSampleRate,
            new float[MonoFloatResampler.TargetSampleRate * 6 / 10]);
        stream.InputFinished();
        while (_recognizer.IsReady(stream))
            _recognizer.Decode(stream);

        // An utterance that produced no words yields nothing at all rather than an empty final:
        // an empty transcript is not a result, and downstream it would clear or re-space a query
        // the user never spoke into.
        var final = SpeechTextFormatter.Format(_recognizer.GetResult(stream).Text);
        if (final.Length > 0)
            yield return new FinalTranscript(final);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _recognizer.Dispose();
        _disposed = true;
    }
}

/// <summary>
/// Makes engine output look like something a person typed into a search box.
///
/// <para>
/// The LibriSpeech-trained Zipformer emits unpunctuated upper case - "FREE UP DISK SPACE" -
/// because that is how its training transcripts are written. Search is case-insensitive, so this
/// is purely about what the user reads back: shouted text in the query box reads as a bug, and it
/// is also not what they would have typed, which matters because the box is editable after
/// dictation.
/// </para>
/// </summary>
internal static class SpeechTextFormatter
{
    public static string Format(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;

        var trimmed = text.Trim();

        // Only fold case when the engine has given us nothing but upper case. A provider that
        // already cases its output - Windows AI Speech does - must be passed through untouched.
        var hasLower = trimmed.Any(char.IsLower);
        return hasLower ? trimmed : trimmed.ToLower(System.Globalization.CultureInfo.CurrentCulture);
    }
}
