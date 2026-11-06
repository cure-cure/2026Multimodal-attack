param(
  [string]$PythonPath,
  [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ConfigPath) { $ConfigPath = Join-Path $ProjectRoot "configs\fusion_config.example.json" }
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

$PythonCommand = $null
$PythonPrefix = @()
if ($PythonPath) {
  $PythonCommand = (Get-Item -LiteralPath $PythonPath -ErrorAction Stop).FullName
} else {
  $Python = Get-Command python -ErrorAction SilentlyContinue
  if ($Python) {
    $PythonCommand = $Python.Source
  } else {
    $Python = Get-Command py -ErrorAction SilentlyContinue
    if ($Python) {
      $PythonCommand = $Python.Source
      $PythonPrefix = @("-3")
    }
  }
}
if (-not $PythonCommand) {
  throw "Python 3.10+ was not found. Install Python and make the python or py command available."
}

& $PythonCommand @PythonPrefix -m mm_jailbreak_guard web `
  --config $ConfigPath `
  --image-evidence (Join-Path $ProjectRoot "examples\handoff\image_evidence.jsonl") `
  --text-evidence (Join-Path $ProjectRoot "examples\handoff\text_evidence.jsonl")
