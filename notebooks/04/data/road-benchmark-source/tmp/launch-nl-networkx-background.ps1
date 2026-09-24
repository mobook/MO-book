$ErrorActionPreference = 'Stop'
$python = 'C:\work\codex\projects\MO-book\tmp\road-preview-env\Scripts\pythonw.exe'
$script = 'C:\work\codex\projects\MO-book\tmp\nl-networkx-background.py'
if (!(Test-Path -LiteralPath $python)) { throw 'Python executable missing' }
if (!(Test-Path -LiteralPath $script)) { throw 'Worker script missing' }
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = ('"{0}" -X utf8 "{1}"' -f $python, $script)
    CurrentDirectory = 'C:\work\codex\projects\MO-book'
    ProcessStartupInformation = $startup
}
if ($result.ReturnValue -ne 0) { throw "Background launch failed: $($result.ReturnValue)" }
$result | Select-Object ProcessId,ReturnValue | ConvertTo-Json
