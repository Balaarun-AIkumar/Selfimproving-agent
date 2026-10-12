param(
    [ValidateSet('agent','chat','improve','test','eval','check-key')][string]$Action = 'agent',
    [ValidateRange(1,65535)][int]$Port = 8000,
    [string]$DataDir = '',
    [switch]$Offline
)
$ErrorActionPreference = 'Stop'
$taskBundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Test-Path -LiteralPath $taskBundledPython) {
    $taskInterpreter = $taskBundledPython
} else {
    $taskPythonCommand = Get-Command python,python3 -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $taskPythonCommand) { throw 'Install Python 3.10 or later, then run this command again.' }
    $taskInterpreter = $taskPythonCommand.Source
}
Push-Location -LiteralPath $PSScriptRoot
$taskModelArgs = @()
if (-not $Offline) { $taskModelArgs = @('--gemini') }
if (-not $DataDir) { $DataDir = if ($Offline) { 'runtime' } else { 'runtime/gemini' } }
try {
    switch ($Action) {
        'agent' { & $taskInterpreter -m scheduler serve --port $Port --data-dir $DataDir @taskModelArgs }
        'chat' { & $taskInterpreter -m scheduler chat @taskModelArgs }
        'improve' { & $taskInterpreter -m scheduler improve @taskModelArgs }
        'test' { & $taskInterpreter -m unittest discover -v }
        'eval' { & $taskInterpreter -m scheduler eval @taskModelArgs }
        'check-key' { & $taskInterpreter -m scheduler check-gemini }
    }
    exit $LASTEXITCODE
} finally { Pop-Location }
