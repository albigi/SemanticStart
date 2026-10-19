using SemanticStart.Core.Speech;
using CpuTimes = SemanticStart.Core.Speech.SpeechInferenceThreads.CpuTimes;

namespace SemanticStart.Tests;

public sealed class SpeechInferenceThreadsTests
{
    [Theory]
    [InlineData(1, null, 1)]
    [InlineData(2, null, 2)]
    [InlineData(3, null, 3)]
    [InlineData(4, null, 4)]
    [InlineData(64, null, 4)]
    [InlineData(1, 0.0, 1)]
    [InlineData(4, 0.0, 4)]
    [InlineData(4, 0.01, 4)]
    [InlineData(4, 0.25, 3)]
    [InlineData(4, 0.5, 2)]
    [InlineData(4, 0.75, 1)]
    [InlineData(8, 0.0, 4)]
    [InlineData(8, 0.5, 4)]
    [InlineData(8, 0.75, 2)]
    [InlineData(8, 0.875, 1)]
    [InlineData(64, 1.0, 1)]
    public void SelectionUsesFourThreadsUnlessProcessorCountOrLoadLimitsCapacity(int processors, double? usage, int expected)
    {
        Assert.Equal(expected, SpeechInferenceThreads.Select(processors, usage));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void InvalidUsageIsRejected(double usage)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => SpeechInferenceThreads.Select(4, usage));
    }

    [Fact]
    public void InvalidProcessorCountIsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => SpeechInferenceThreads.Select(0));
    }

    [Fact]
    public void UsageUsesDeltasAndIncludesIdleOnlyOnce()
    {
        var usage = SpeechInferenceThreads.CalculateUsage(new CpuTimes(100, 200, 300), new CpuTimes(160, 280, 340));

        Assert.Equal(0.5, usage);
    }

    [Theory]
    [InlineData(100, 200, 300)]
    [InlineData(99, 280, 340)]
    [InlineData(160, 199, 340)]
    [InlineData(160, 280, 299)]
    [InlineData(300, 280, 340)]
    public void InvalidTimingSamplesHaveNoUsage(int idle, int kernel, int user)
    {
        Assert.Null(SpeechInferenceThreads.CalculateUsage(
            new CpuTimes(100, 200, 300), new CpuTimes((ulong)idle, (ulong)kernel, (ulong)user)));
    }

    [Fact]
    public async Task SelectionReadsWindowsCountersWithoutLoadingAModel()
    {
        var threads = await SpeechInferenceThreads.SelectAsync();

        Assert.InRange(threads, 1, SpeechInferenceThreads.Select(Environment.ProcessorCount));
    }

    [Fact]
    public async Task SelectionHonorsCancellation()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => SpeechInferenceThreads.SelectAsync(cancellation.Token));
    }
}
