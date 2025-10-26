Param(
    [switch]$Force
)

$dbs = Get-ChildItem -Recurse -Filter *.db -ErrorAction SilentlyContinue
if (-not $dbs -or $dbs.Count -eq 0) {
    Write-Host "No .db files found under $(Get-Location)"
    exit 0
}

Write-Host "Found the following .db files:" -ForegroundColor Cyan
$dbs | ForEach-Object { Write-Host " - $($_.FullName)" }

if (-not $Force) {
    $resp = Read-Host "Delete ALL of the above files? (y/N)"
    if ($resp -ne 'y' -and $resp -ne 'Y') { Write-Host "Aborted."; exit 1 }
}

$dbs | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host "Deleted $($dbs.Count) .db file(s)." -ForegroundColor Green

