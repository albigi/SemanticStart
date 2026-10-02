using SemanticStart.Core.Speech;

namespace SemanticStart.Tests;

public sealed class SpeechRecognitionConfigTests
{
    [Fact]
    public void RecognitionUsesFourBeamPathsAndKeepsSileroEndpointing()
    {
        var files = new SpeechModelFiles("encoder", "decoder", "joiner", "tokens", "vad");
        var config = SherpaOnnxSpeechTranscriber.CreateConfig(files);

        Assert.Equal("modified_beam_search", config.DecodingMethod);
        Assert.Equal(4, config.MaxActivePaths);
        Assert.Equal(0, config.EnableEndpoint);
        Assert.Equal(16_000, config.FeatConfig.SampleRate);
        Assert.Equal(files.EncoderPath, config.ModelConfig.Transducer.Encoder);
    }
}
