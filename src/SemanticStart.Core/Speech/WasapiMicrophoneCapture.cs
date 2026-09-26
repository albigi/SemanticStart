using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Channels;
using NAudio.CoreAudioApi;
using NAudio.Wave;

namespace SemanticStart.Core.Speech;

/// <summary>
/// The default recording device, opened through WASAPI in shared mode and delivered as 16 kHz
/// mono float.
///
/// <para>
/// Shared mode, not exclusive: exclusive mode would take the microphone away from whatever else
/// is using it, and a launcher that breaks the call you are on to let you search is not a trade
/// anyone would make. The cost is that the device dictates the format - typically 48 kHz stereo
/// float - which <see cref="MonoFloatResampler"/> converts.
/// </para>
/// <para>
/// The buffer is asked to be short and the client is event-driven, because this audio is on the
/// latency path the whole feature is judged by: every millisecond of buffering is a millisecond
/// added to both the first partial result and the final one. Windows rounds the request up to the
/// engine's own period, so this is a floor rather than a guarantee.
/// </para>
/// <para>
/// Every way this can fail ends in a <see cref="MicrophoneUnavailableException"/> carrying a
/// reason. Dictation that silently produces no words is indistinguishable from dictation that is
/// broken, and the most common cause - the global desktop-app microphone switch being off - is
/// invisible unless it is named.
/// </para>
/// </summary>
public sealed class WasapiMicrophoneCapture : IAudioCaptureSource
{
    /// <summary>
    /// Requested capture period. 20 ms is short enough not to dominate the latency budget and
    /// long enough to survive an ordinary scheduling hiccup without dropping packets.
    /// </summary>
    private const int BufferMilliseconds = 20;

    private const int AccessDenied = unchecked((int)0x80070005);
    private const int AudioClientDeviceInvalidated = unchecked((int)0x88890004);
    private const int NotFound = unchecked((int)0x80070490);

    private bool _disposed;

    public int SampleRate => MonoFloatResampler.TargetSampleRate;

    public async IAsyncEnumerable<ReadOnlyMemory<float>> CaptureAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        using var activity = SpeechDiagnostics.StartActivity(SpeechDiagnostics.CaptureActivity);

        using var device = OpenDefaultDevice();
        using var capture = new WasapiCapture(device, useEventSync: true, audioBufferMillisecondsLength: BufferMilliseconds);

        var format = capture.WaveFormat;
        var resampler = new MonoFloatResampler(format.SampleRate, format.Channels);

        // The device's own mix format, not ours: when capture misbehaves this is the first thing
        // anyone reading a trace needs, and it is different on every machine.
        activity?.SetTag("audio.device.sample_rate", format.SampleRate);
        activity?.SetTag("audio.device.channels", format.Channels);
        activity?.SetTag("audio.device.bits_per_sample", format.BitsPerSample);
        activity?.SetTag("audio.device.encoding", format.Encoding.ToString());

        var captured = 0L;
        var dropped = 0L;

        // Dropping the oldest buffer beats blocking the audio thread: WASAPI's callback runs on a
        // real-time thread and stalling it glitches capture for every app on the machine. A
        // consumer far enough behind to hit this bound has already lost the utterance anyway.
        //
        // A dropped buffer is still audio the user spoke that the transcriber never saw, and it
        // cannot be thrown - the audio thread has nobody to throw to - so it is counted instead and
        // reported with the trace, so a transcript with a hole in it has an explanation.
        var channel = Channel.CreateBounded<ReadOnlyMemory<float>>(
            new BoundedChannelOptions(128)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleWriter = true,
                SingleReader = true,
            },
            itemDropped: _ => Interlocked.Increment(ref dropped));

        void OnDataAvailable(object? sender, WaveInEventArgs args)
        {
            if (args.BytesRecorded <= 0)
                return;

            var samples = ToFloatSamples(args.Buffer, args.BytesRecorded, format);
            var converted = resampler.Resample(samples);
            if (converted.Length == 0)
                return;

            captured += converted.Length;
            channel.Writer.TryWrite(converted);
        }

        void OnRecordingStopped(object? sender, StoppedEventArgs args)
        {
            // NAudio catches everything the capture thread throws, including anything raised from
            // the handler above, and delivers it here instead of anywhere a caller can see it.
            // Completing the channel with it is what turns it back into a thrown exception.
            if (args.Exception is { } ex)
                SpeechDiagnostics.ReportFailure("capture.thread", "The microphone capture thread stopped with an error.", ex);

            channel.Writer.TryComplete(args.Exception is { } failure ? Describe(failure) : null);
        }

        capture.DataAvailable += OnDataAvailable;
        capture.RecordingStopped += OnRecordingStopped;

        try
        {
            try
            {
                capture.StartRecording();
            }
            catch (Exception ex)
            {
                throw Describe(ex);
            }

            await foreach (var buffer in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
                yield return buffer;
        }
        finally
        {
            capture.DataAvailable -= OnDataAvailable;
            capture.RecordingStopped -= OnRecordingStopped;

            activity?.SetTag("audio.samples_captured", captured);
            activity?.SetTag("audio.buffers_dropped", Interlocked.Read(ref dropped));

            if (Interlocked.Read(ref dropped) > 0)
            {
                SpeechDiagnostics.Report(
                    "capture.dropped",
                    $"The transcriber fell behind the microphone and {Interlocked.Read(ref dropped)} buffer(s) were dropped.");
            }

            try
            {
                capture.StopRecording();
            }
            catch (Exception ex)
            {
                // Stopping a device that has already gone away does not fail the session - it is
                // ending either way - but it is still a native call that failed, and a COM error
                // that nobody ever hears about is how a whole class of device bugs stays invisible.
                SpeechDiagnostics.ReportFailure("capture.stop", "Stopping the microphone failed.", ex);
            }
        }
    }

    public void Dispose() => _disposed = true;

    private static MMDevice OpenDefaultDevice()
    {
        try
        {
            using var enumerator = new MMDeviceEnumerator();

            // The communications endpoint rather than the multimedia one: it is what Windows
            // points voice at, so it follows the headset the user actually speaks into, and it
            // carries the endpoint's own voice processing.
            if (!enumerator.HasDefaultAudioEndpoint(DataFlow.Capture, Role.Communications))
            {
                throw new MicrophoneUnavailableException(
                    MicrophoneFailure.NoDevice,
                    MicrophoneUnavailableException.DescribeFailure(MicrophoneFailure.NoDevice));
            }

            return enumerator.GetDefaultAudioEndpoint(DataFlow.Capture, Role.Communications);
        }
        catch (MicrophoneUnavailableException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw Describe(ex);
        }
    }

    /// <summary>
    /// Turns a WASAPI failure into something the user can act on. The HRESULTs are the ones that
    /// distinguish "you have not allowed this" from "there is nothing to record with", which are
    /// the two cases with different fixes.
    /// </summary>
    internal static MicrophoneUnavailableException Describe(Exception exception)
    {
        if (exception is MicrophoneUnavailableException microphone)
            return microphone;

        var failure = exception switch
        {
            UnauthorizedAccessException => MicrophoneFailure.AccessDenied,
            COMException { ErrorCode: AccessDenied } => MicrophoneFailure.AccessDenied,
            COMException { ErrorCode: NotFound } => MicrophoneFailure.NoDevice,
            COMException { ErrorCode: AudioClientDeviceInvalidated } => MicrophoneFailure.NoDevice,
            ArgumentException => MicrophoneFailure.NoDevice,
            _ => MicrophoneFailure.DeviceError,
        };

        return new MicrophoneUnavailableException(
            failure,
            MicrophoneUnavailableException.DescribeFailure(failure),
            exception);
    }

    /// <summary>
    /// Reads one capture buffer as floats. Shared mode hands over the audio engine's mix format,
    /// which is float on every machine seen so far but is not promised to be, so 16-bit PCM is
    /// handled rather than assumed away.
    /// </summary>
    internal static float[] ToFloatSamples(byte[] buffer, int bytesRecorded, WaveFormat format)
    {
        if (format.Encoding == WaveFormatEncoding.IeeeFloat && format.BitsPerSample == 32)
        {
            var samples = new float[bytesRecorded / sizeof(float)];
            Buffer.BlockCopy(buffer, 0, samples, 0, samples.Length * sizeof(float));
            return samples;
        }

        if (format.BitsPerSample == 16)
        {
            var samples = new float[bytesRecorded / sizeof(short)];
            for (var i = 0; i < samples.Length; i++)
                samples[i] = BitConverter.ToInt16(buffer, i * sizeof(short)) / 32768f;

            return samples;
        }

        throw new MicrophoneUnavailableException(
            MicrophoneFailure.DeviceError,
            $"The microphone's format ({format.Encoding}, {format.BitsPerSample}-bit) is not supported.");
    }
}
