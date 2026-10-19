using System.ComponentModel;
using System.Runtime.InteropServices;

namespace SemanticStart.Core.Speech;

internal static class SpeechInferenceThreads
{
    public static async Task<int> SelectAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var processors = Environment.ProcessorCount;
        double? usage = null;

        // GetSystemTimes covers only the calling processor group on machines with >64 CPUs.
        if (OperatingSystem.IsWindows() && processors > 1 && GetActiveProcessorGroupCount() == 1)
        {
            try
            {
                var before = ReadTimes();
                await Task.Delay(200, cancellationToken).ConfigureAwait(false);
                usage = CalculateUsage(before, ReadTimes());
                if (usage is null)
                    SpeechDiagnostics.ReportFailure("cpu-load", "CPU timing sample was invalid; using processor count.");
            }
            catch (Win32Exception ex)
            {
                SpeechDiagnostics.ReportFailure("cpu-load", "Could not read CPU load; using processor count.", ex);
            }
        }

        var threads = Select(processors, usage);
        SpeechDiagnostics.Report("inference-threads",
            $"Using {threads} speech inference thread(s); processors={processors}, " +
            $"CPU load={(usage is { } value ? $"{value:P0}" : "unavailable")}.");
        return threads;
    }

    internal static int Select(int processors, double? usage = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(processors, 1);
        if (usage is { } value && (!double.IsFinite(value) || value is < 0 or > 1))
            throw new ArgumentOutOfRangeException(nameof(usage));

        // Four threads is the performance baseline; back off only when capacity is busy.
        var maximum = Math.Min(processors, 4);
        return usage is { } busy
            ? Math.Clamp((int)Math.Ceiling(processors * (1 - busy)), 1, maximum)
            : maximum;
    }

    internal static double? CalculateUsage(CpuTimes before, CpuTimes after)
    {
        if (after.Idle < before.Idle || after.Kernel < before.Kernel || after.User < before.User)
            return null;

        // Windows includes idle time in kernel time, so it must not be added again.
        var total = (double)(after.Kernel - before.Kernel) + (after.User - before.User);
        var idle = after.Idle - before.Idle;
        return total > 0 && idle <= total ? 1 - idle / total : null;
    }

    private static CpuTimes ReadTimes()
    {
        if (!GetSystemTimes(out var idle, out var kernel, out var user))
            throw new Win32Exception(Marshal.GetLastWin32Error());
        return new CpuTimes(idle, kernel, user);
    }

    internal readonly record struct CpuTimes(ulong Idle, ulong Kernel, ulong User);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetSystemTimes(out ulong idle, out ulong kernel, out ulong user);

    [DllImport("kernel32.dll")]
    private static extern ushort GetActiveProcessorGroupCount();
}
