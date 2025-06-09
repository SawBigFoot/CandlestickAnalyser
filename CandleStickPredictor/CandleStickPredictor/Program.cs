using System.Diagnostics;
using System.Net;
using System.Text;
using System.Threading.Tasks;

class Program
{
    private static Process? pythonProcess;
    private static HttpListener? server;
    private static bool isRunning = true;

    static async Task Main()
    {
        // Start the HTTP server
        await StartServer();

        // Start the Python bot
        StartPythonBot();

        // Keep the application running
        while (isRunning)
        {
            await Task.Delay(1000);
        }
    }

    static async Task StartServer()
    {
        server = new HttpListener();
        server.Prefixes.Add("http://localhost:8080/");
        server.Start();
        Console.WriteLine("Server started at http://localhost:8080/");

        // Handle requests in a separate task
        _ = Task.Run(async () =>
        {
            while (isRunning)
            {
                try
                {
                    var context = await server.GetContextAsync();
                    var response = context.Response;
                    var buffer = Encoding.UTF8.GetBytes("Server is running...");
                    response.ContentLength64 = buffer.Length;
                    await response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
                    response.Close();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Server error: {ex.Message}");
                }
            }
        });
    }

    static void StartPythonBot()
    {
        string pythonPath = "python3";
        string scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "bot.py");

        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = $"\"{scriptPath}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        pythonProcess = new Process { StartInfo = psi };
        pythonProcess.OutputDataReceived += (sender, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
                Console.WriteLine($"Bot: {e.Data}");
        };
        pythonProcess.ErrorDataReceived += (sender, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
                Console.WriteLine($"Bot Error: {e.Data}");
        };

        pythonProcess.Start();
        pythonProcess.BeginOutputReadLine();
        pythonProcess.BeginErrorReadLine();

        Console.WriteLine("Python bot started");
    }

    static void Cleanup()
    {
        isRunning = false;
        if (pythonProcess != null && !pythonProcess.HasExited)
        {
            pythonProcess.Kill();
            pythonProcess.Dispose();
        }
        if (server != null && server.IsListening)
        {
            server.Stop();
            server.Close();
        }
    }
}
