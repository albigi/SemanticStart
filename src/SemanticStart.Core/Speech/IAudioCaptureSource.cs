namespace SemanticStart.Core.Speech;

/// <summary>
/// The microphone, as everything above it needs to see it: a stream of 16 kHz mono float buffers
/// and nothing else.
///
/// <para>
/// An interface rather than the WASAPI class directly, because every consumer of audio in this
/// feature - endpointing, the transcriber, the level meter, and their tests - can be exercised
/// from recorded samples, and a build server has no microphone.
/// </para>
/// </summary>
public interface IAudioCaptureSource : IDisposable
{
    /// <summary>Sample rate of the buffers yielded by <see cref="CaptureAsync"/>.</summary>
    int SampleRate { get; }

    /// <summary>
    /// Opens the device and yields captured audio until cancelled. Failures to open are thrown as
    /// <see cref="MicrophoneUnavailableException"/> rather than ending the stream quietly: a
    /// dictation session that produces no words has to be able to say why.
    /// </summary>
    IAsyncEnumerable<ReadOnlyMemory<float>> CaptureAsync(CancellationToken cancellationToken = default);
}

/// <summary>Why the microphone could not be opened, in the terms the user can act on.</summary>
public enum MicrophoneFailure
{
    /// <summary>No capture device is present or enabled.</summary>
    NoDevice,

    /// <summary>
    /// Windows refused access. For an unpackaged desktop app there is no per-app microphone
    /// permission on current stable Windows, only the global switch, so the message names it.
    /// </summary>
    AccessDenied,

    /// <summary>The device is there and permitted, but the audio client failed.</summary>
    DeviceError,
}

public sealed class MicrophoneUnavailableException : Exception
{
    /// <summary>
    /// The exact wording of the Windows setting that governs microphone access for desktop apps.
    /// Named in full because "check your privacy settings" sends a user to a page with a dozen
    /// switches, and the one that matters is below a list of Store apps that do not include us.
    /// </summary>
    public const string DesktopMicrophoneSettingName = "Let desktop apps access your microphone";

    public MicrophoneUnavailableException(MicrophoneFailure failure, string message, Exception? innerException = null)
        : base(message, innerException)
    {
        Failure = failure;
    }

    public MicrophoneFailure Failure { get; }

    /// <summary>The message to show in the overlay for a given failure.</summary>
    public static string DescribeFailure(MicrophoneFailure failure) => failure switch
    {
        MicrophoneFailure.NoDevice =>
            "No microphone was found. Connect one, or enable it under Settings > System > Sound > Input.",
        MicrophoneFailure.AccessDenied =>
            $"Windows blocked microphone access. Turn on \"{DesktopMicrophoneSettingName}\" under Settings > Privacy & security > Microphone.",
        _ => "The microphone could not be started; see the log for details.",
    };
}
