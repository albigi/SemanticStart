using SemanticStart.Core.Speech;

namespace SemanticStart.Tests;

public sealed class SpeechRecognitionConfigTests
{
    [Fact]
    public void StreamingRecognitionKeepsAlternativeWordSequences()
    {
        var files = new SpeechModelFiles("encoder", "decoder", "joiner", "tokens", "vad");

        var config = SherpaOnnxSpeechTranscriber.CreateConfig(files);

        Assert.Equal("modified_beam_search", config.DecodingMethod);
        Assert.Equal(4, config.MaxActivePaths);
        Assert.Equal(2, config.ModelConfig.NumThreads);
        Assert.Equal(0, config.EnableEndpoint);
        Assert.Equal(16000, config.FeatConfig.SampleRate);
        Assert.Equal(files.EncoderPath, config.ModelConfig.Transducer.Encoder);
        Assert.Equal(files.DecoderPath, config.ModelConfig.Transducer.Decoder);
        Assert.Equal(files.JoinerPath, config.ModelConfig.Transducer.Joiner);
        Assert.Equal(files.TokensPath, config.ModelConfig.Tokens);
    }
}
