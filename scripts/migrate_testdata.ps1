Param(
    [string]$RepoRoot = (Resolve-Path "$PSScriptRoot/.."),
    [string]$Dest = (Join-Path (Resolve-Path "$PSScriptRoot/..") "testdata/public")
)

Write-Host "RepoRoot = $RepoRoot"
Write-Host "Dest     = $Dest"

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

function Move-IfExists($path, $destDir) {
    if (Test-Path $path) {
        Write-Host "Moving" $path "->" $destDir
        New-Item -ItemType Directory -Force -Path $destDir | Out-Null
        Move-Item -LiteralPath $path -Destination $destDir -Force
    } else {
        Write-Host "Skip (not found):" $path
    }
}

# Embedder demo small samples
$eds = Join-Path $RepoRoot "tools/embedder-demo/testdata/small"
Move-IfExists (Join-Path $eds "one_col_10_records_with_header.xlsx") $Dest
Move-IfExists (Join-Path $eds "one_col_300_records_with_header.xlsx") $Dest
Move-IfExists (Join-Path $eds "one_col_10_records_with_header_utf8.csv") $Dest
Move-IfExists (Join-Path $eds "one_col_10_records_with_header_sjis.csv") $Dest
Move-IfExists (Join-Path $eds "one_col_10_records_with_header_utf8bom.csv") $Dest
Move-IfExists (Join-Path $eds "one_col_300_records_with_header_utf8bom.csv") $Dest

# PDF block viewer samples (move all files in its testdata)
$pbv = Join-Path $RepoRoot "tools/pdf-block-viewer/testdata"
if (Test-Path $pbv) {
    Get-ChildItem -LiteralPath $pbv -File | ForEach-Object {
        Write-Host "Moving" $_.FullName "->" $Dest
        Move-Item -LiteralPath $_.FullName -Destination $Dest -Force
    }
}

Write-Host "Done. Review moved files under" $Dest
