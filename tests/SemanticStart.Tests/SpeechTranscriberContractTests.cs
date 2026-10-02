using SemanticStart.Core.Speech;

namespace SemanticStart.Tests;

/// <summary>
/// What the rest of the app is allowed to assume about any transcriber, asserted against a fake so
/// the contract is pinned without loading a model. Partial and final are different kinds of claim
/// - a partial may still be revised - so the order they arrive in is the contract, not a detail.
/// </summary>
public sealed class SpeechTranscriberContractTests
{
    private static readonly SpeechProviderMetadata Metadata = new(
        Id: "test",
        DisplayName: "Test engine",
        ModelId: "test-model",
        Language: "en-US",
        SupportsPartialResults: true,
        RequiresModelDownload: false,
        ApproximateDownloadBytes: 0);

    [Fact]
    public async Task PartialsArriveInOrderAndTheLastItemIsTheFinalTranscript()
    {
        using var transcriber = new TestSpeechTranscriber(
            Metadata,
            new PartialTranscript("visual"),
            new PartialTranscript("visual studio"),
            new FinalTranscript("visual studio code"));

        var received = new List<SpeechTranscript>();
        await foreach (var transcript in transcriber.TranscribeAsync(Audio(3)))
            received.Add(transcript);

        Assert.Equal(["visual", "visual studio", "visual studio code"], received.Select(t => t.Text));
        Assert.Equal([false, false, true], received.Select(t => t.IsFinal));
        Assert.IsType<FinalTranscript>(received[^1]);
    }

    /// <summary>
    /// The stream ends when the audio does. A transcriber that kept the enumeration open would
    /// hang the overlay on the one path - endpoint reached, search now - the feature exists for.
    /// </summary>
    [Fact]
    public async Task TheEnumerationCompletesWhenTheAudioDoes()
    {
        using var transcriber = new TestSpeechTranscriber(Metadata, new FinalTranscript("notepad"));

        var received = new List<SpeechTranscript>();
        await foreach (var transcript in transcriber.TranscribeAsync(Audio(2)))
            received.Add(transcript);

        var only = Assert.Single(received);
        Assert.Equal("notepad", only.Text);
        Assert.True(only.IsFinal);

        // Every buffer the caller produced was consumed, in order, rather than the transcriber
        // stopping at the first transcript it had.
        Assert.Equal(2, transcriber.Received.Count);
        Assert.Equal([0f, 0f], transcriber.Received[0]);
    }

    /// <summary>
    /// An utterance with nothing audible in it still has to terminate. A transcriber with nothing
    /// to say ends the stream rather than leaving the caller waiting for a transcript that is
    /// never coming.
    /// </summary>
    [Fact]
    public async Task ASilentUtteranceYieldsNothingAndStillCompletes()
    {
        using var transcriber = new TestSpeechTranscriber(Metadata);

        var received = new List<SpeechTranscript>();
        await foreach (var transcript in transcriber.TranscribeAsync(Audio(1)))
            received.Add(transcript);

        Assert.Empty(received);
    }

    [Fact]
    public async Task CancellationStopsTheStream()
    {
        using var cancellation = new CancellationTokenSource();
        using var transcriber = new TestSpeechTranscriber(
            Metadata,
            new PartialTranscript("visual"),
            new FinalTranscript("visual studio"));

        var received = new List<SpeechTranscript>();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var transcript in transcriber.TranscribeAsync(Audio(4), cancellation.Token))
            {
                received.Add(transcript);
                await cancellation.CancelAsync();
            }
        });

        Assert.Single(received);
    }

    /// <summary>16 kHz mono buffers, which is the only format the interface accepts.</summary>
    private static async IAsyncEnumerable<ReadOnlyMemory<float>> Audio(int buffers)
    {
        for (var i = 0; i < buffers; i++)
        {
            await Task.Yield();
            yield return new float[2];
        }
    }
}
