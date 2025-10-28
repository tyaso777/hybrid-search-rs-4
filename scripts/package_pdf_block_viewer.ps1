param(
  [string]$Configuration = "release",
  [string]$OutDir = "dist/pdf-block-viewer"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "[1/4] Building pdf-block-viewer ($Configuration)..."
cargo build --$Configuration -p pdf-block-viewer | Out-Host

$exe = Join-Path -Path (Resolve-Path .).Path -ChildPath "target/$Configuration/pdf-block-viewer.exe"
if (-not (Test-Path $exe)) { throw "Executable not found: $exe" }

Write-Host "[2/4] Preparing output directory: $OutDir"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "[3/4] Copying executable"
Copy-Item -Force $exe (Join-Path $OutDir 'pdf-block-viewer.exe')

# Try to include a PDFium DLL if one can be located locally.
function Find-PdfiumDll {
  if ($env:PDFIUM_DLL_PATH -and (Test-Path $env:PDFIUM_DLL_PATH)) { return (Resolve-Path $env:PDFIUM_DLL_PATH).Path }
  $candidates = @(
    'file-chunker/bin/pdfium-win-x64/bin/pdfium.dll',
    'file-chunker/bin/pdfium-win-x64/pdfium.dll',
    'file-chunker/bin/pdfium.dll'
  )
  foreach ($rel in $candidates) {
    $p = Join-Path -Path (Resolve-Path .).Path -ChildPath $rel
    if (Test-Path $p) { return $p }
  }
  return $null
}

$pdfium = Find-PdfiumDll
if ($pdfium) {
  Write-Host "[4/4] Bundling PDFium: $pdfium"
  Copy-Item -Force $pdfium (Join-Path $OutDir 'pdfium.dll')
} else {
  Write-Warning "PDFium DLL not found. The viewer will fall back to pure-Rust or require PDFium on PATH."
}

# Minimal README for end users
@"
PDF Block Viewer (portable)
===========================

Run: pdf-block-viewer.exe

Notes
- If pdfium.dll is present in this folder, the viewer uses the PDFium backend automatically.
- If not present, ensure PDFium is available on PATH or set PDFIUM_DLL_PATH to its location.
- Without PDFium, the app can still run using the pure-Rust fallback (limited extraction quality).

Environment variables (optional)
- PDFIUM_DLL_PATH: absolute path to pdfium.dll
- PDFIUM_DIR: folder containing pdfium.dll
"@ | Set-Content -Encoding UTF8 (Join-Path $OutDir 'README.txt')

Write-Host "Done -> $OutDir"

