using SemanticStart.Core.Speech;

namespace SemanticStart.Tests;

public sealed class SpeechRecognitionConfigTests
{
    /// <summary>
    /// The native Parakeet Unified recognizer only implements greedy decoding; asking it for
    /// <c>modified_beam_search</c> is a fatal (process-exiting) error on the native side rather
    /// than a recoverable one, so this is pinned as a regression guard rather than a preference.
    /// </summary>
    [Fact]
    public void StreamingRecognitionUsesGreedyDecodingWithEndpointingDisabled()
    {
        var files = new SpeechModelFiles("encoder", "decoder", "joiner", "tokens", "vad");

        var config = SherpaOnnxSpeechTranscriber.CreateConfig(files, numThreads: 4);

        Assert.Equal("greedy_search", config.DecodingMethod);
        Assert.Equal(4, config.ModelConfig.NumThreads);
        Assert.Equal(0, config.EnableEndpoint);
        Assert.Equal(16000, config.FeatConfig.SampleRate);
        Assert.Equal(128, config.FeatConfig.FeatureDim);
        Assert.Equal(files.EncoderPath, config.ModelConfig.Transducer.Encoder);
        Assert.Equal(files.DecoderPath, config.ModelConfig.Transducer.Decoder);
        Assert.Equal(files.JoinerPath, config.ModelConfig.Transducer.Joiner);
        Assert.Equal(files.TokensPath, config.ModelConfig.Tokens);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void SelectedThreadCountReachesNativeConfiguration(int threads)
    {
        var files = new SpeechModelFiles("encoder", "decoder", "joiner", "tokens", "vad");

        Assert.Equal(threads, SherpaOnnxSpeechTranscriber.CreateConfig(files, numThreads: threads).ModelConfig.NumThreads);
    }

    [Fact]
    public void DefaultThreadCountUsesProcessorCountFallback()
    {
        var files = new SpeechModelFiles("encoder", "decoder", "joiner", "tokens", "vad");

        Assert.Equal(Math.Min(Environment.ProcessorCount, 4),
            SherpaOnnxSpeechTranscriber.CreateConfig(files).ModelConfig.NumThreads);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(5)]
    public void InvalidThreadCountsAreRejected(int threads)
    {
        var files = new SpeechModelFiles("encoder", "decoder", "joiner", "tokens", "vad");

        Assert.Throws<ArgumentOutOfRangeException>(
            () => SherpaOnnxSpeechTranscriber.CreateConfig(files, numThreads: threads));
    }
}
