# Generate vulnerability and license reports for the workspace
#
# Usage (from repo root):
#   pwsh -File scripts/generate_reports.ps1
#   # or on Windows PowerShell
#   powershell -ExecutionPolicy Bypass -File scripts/generate_reports.ps1
#
# Options:
#   -InstallTools   Install cargo-audit and cargo-license if missing
#   -OutDir <path>  Output directory (default: reports)
#   -IncludeDevDeps Include dev-dependencies in license report (default: false)
#   -NoAllFeatures  Do not pass --all-features to cargo license (default: uses all features)

[CmdletBinding()]
param(
  [switch] $InstallTools,
  [string] $OutDir = 'reports',
  [switch] $IncludeDevDeps,
  [switch] $NoAllFeatures
)

$ErrorActionPreference = 'Stop'

function Ensure-Tool {
  param(
    [Parameter(Mandatory=$true)] [string] $Bin,
    [Parameter(Mandatory=$true)] [string] $Crate
  )
  if (-not (Get-Command $Bin -ErrorAction SilentlyContinue)) {
    Write-Host "Installing $Crate..."
    cargo install $Crate --locked
  }
}

Write-Host '=== Generating reports ==='

if ($InstallTools) {
  Ensure-Tool -Bin 'cargo-audit' -Crate 'cargo-audit'
  Ensure-Tool -Bin 'cargo-license' -Crate 'cargo-license'
  Ensure-Tool -Bin 'cargo-about' -Crate 'cargo-about'
}

if (-not (Test-Path $OutDir)) {
  New-Item -ItemType Directory -Path $OutDir | Out-Null
}

# 1) Vulnerability report (cargo-audit)
Write-Host '-> Running cargo audit (text + JSON)'
& cargo audit --json | Set-Content -Encoding UTF8 (Join-Path $OutDir 'cargo-audit.json')
& cargo audit | Set-Content -Encoding UTF8 (Join-Path $OutDir 'cargo-audit.txt')

# 2) License report (aggregate workspace members)
Write-Host '-> Collecting license info across workspace'
$metaJson = cargo -q metadata --format-version 1 | ConvertFrom-Json
$pkgMap = @{}
foreach ($p in $metaJson.packages) { $pkgMap[$p.id] = $p }
$members = $metaJson.workspace_members
if (-not $members -or $members.Count -eq 0) {
  throw 'No workspace members found via cargo metadata'
}

$licenseArgs = @()
if (-not $NoAllFeatures) { $licenseArgs += '--all-features' }
if (-not $IncludeDevDeps) { $licenseArgs += '--avoid-dev-deps' }
$licenseArgs += '--json'

$seen = @{}
$all = @()
foreach ($id in $members) {
  $manifest = $pkgMap[$id].manifest_path
  Write-Host ("   - cargo license for {0}" -f $manifest)
  $out = & cargo license --manifest-path $manifest @licenseArgs | ConvertFrom-Json
  foreach ($item in $out) {
    $key = ($item.name + '@' + $item.version)
    if (-not $seen.ContainsKey($key)) {
      $seen[$key] = $true
      $all += $item
    }
  }
}

# Write JSON
$licenseJsonPath = Join-Path $OutDir 'license.json'
$all | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $licenseJsonPath

# Write text summary
$groups = $all | Group-Object -Property license | Sort-Object -Property Count -Descending
$sb = New-Object System.Text.StringBuilder
[void] $sb.AppendLine('License Summary (unique crates across workspace)')
foreach ($g in $groups) { [void] $sb.AppendLine(("{0}`t{1}" -f $g.Count, $g.Name)) }
[void] $sb.AppendLine('')
[void] $sb.AppendLine('Crates:')
foreach ($it in ($all | Sort-Object name, version)) { [void] $sb.AppendLine(("{0} {1}`t{2}" -f $it.name, $it.version, $it.license)) }

$licenseTxtPath = Join-Path $OutDir 'license.txt'
$sb.ToString() | Set-Content -Encoding UTF8 $licenseTxtPath

# 3) THIRD-PARTY-NOTICES (exclude workspace crates)
Write-Host '-> Generating THIRD-PARTY-NOTICES.txt'
$wsNames = @()
foreach ($id in $members) { $wsNames += $pkgMap[$id].name }
$thirdParty = $all | Where-Object { $wsNames -notcontains $_.name }

$tpPath = Join-Path $OutDir 'THIRD-PARTY-NOTICES.txt'
$sb2 = New-Object System.Text.StringBuilder
[void] $sb2.AppendLine('Third-Party Notices')
[void] $sb2.AppendLine('')
[void] $sb2.AppendLine(("Generated: {0:yyyy-MM-dd HH:mm:ss} UTC" -f ([DateTime]::UtcNow)))
[void] $sb2.AppendLine('This file lists third-party Rust crates used by this project, with their licenses and repositories.')
[void] $sb2.AppendLine('This is an informational summary; refer to each crate for authoritative license text.')
[void] $sb2.AppendLine('')

# Summary by license (third-party only)
$tpGroups = $thirdParty | Group-Object -Property license | Sort-Object -Property Count -Descending
[void] $sb2.AppendLine('Summary by license (third-party only)')
foreach ($g in $tpGroups) { [void] $sb2.AppendLine(("{0}`t{1}" -f $g.Count, $g.Name)) }
[void] $sb2.AppendLine('')

[void] $sb2.AppendLine('Third-party crates:')
foreach ($it in ($thirdParty | Sort-Object name, version)) {
  $repo = $it.repository; if (-not $repo) { $repo = '' }
  $lic = $it.license; if (-not $lic) { $lic = 'UNKNOWN' }
  [void] $sb2.AppendLine(("- {0} {1}  [{2}]  {3}" -f $it.name, $it.version, $lic, $repo))
}

$sb2.ToString() | Set-Content -Encoding UTF8 $tpPath

Write-Host '=== Done ==='
Write-Host ("- Vulnerabilities: {0}" -f (Join-Path $OutDir 'cargo-audit.txt'))
Write-Host ("- Vulnerabilities (JSON): {0}" -f (Join-Path $OutDir 'cargo-audit.json'))
Write-Host ("- Licenses: {0}" -f $licenseTxtPath)
Write-Host ("- Licenses (JSON): {0}" -f $licenseJsonPath)
Write-Host ("- Third-party notices: {0}" -f $tpPath)

# 4) cargo-about: Generate consolidated third-party licenses with license texts
if (Get-Command cargo-about -ErrorAction SilentlyContinue) {
  Write-Host '-> Generating cargo-about (JSON + notices)'
  $aboutJson = Join-Path $OutDir 'about.json'
  cargo about generate --format json --all-features --workspace -o $aboutJson | Out-Null

  # Build THIRD-PARTY-LICENSES.txt by grouping crates by license id and embedding license text
  $aj = Get-Content $aboutJson -Raw | ConvertFrom-Json
  $crates = $aj.crates
  $overview = $aj.overview
  $thirdPartyCrates = $crates | Where-Object { $_.package.source }

  $tpl = New-Object System.Text.StringBuilder
  [void] $tpl.AppendLine('THIRD-PARTY LICENSES (cargo-about)')
  [void] $tpl.AppendLine('')
  [void] $tpl.AppendLine(("Generated: {0:yyyy-MM-dd HH:mm:ss} UTC" -f ([DateTime]::UtcNow)))
  [void] $tpl.AppendLine('This file contains license texts for third-party crates used by this project.')
  [void] $tpl.AppendLine('Workspace/local crates are excluded.')
  [void] $tpl.AppendLine('')

  foreach ($ov in $overview) {
    $licId = $ov.id
    $licName = $ov.name
    [void] $tpl.AppendLine(("=== {0} ({1}) ===" -f $licName, $licId))
    [void] $tpl.AppendLine('Used by:')
    $used = $thirdPartyCrates | Where-Object { $_.license -eq $licId }
    foreach ($c in ($used | Sort-Object { $_.package.name }, { $_.package.version })) {
      $repo = $c.package.repository
      if (-not $repo) { $repo = '' }
      [void] $tpl.AppendLine(("- {0} {1}  {2}" -f $c.package.name, $c.package.version, $repo))
    }
    [void] $tpl.AppendLine('')
    [void] $tpl.AppendLine('License text:')
    [void] $tpl.AppendLine(('-'*79))
    if ($ov.text) { [void] $tpl.AppendLine($ov.text) } else { [void] $tpl.AppendLine('N/A') }
    [void] $tpl.AppendLine(('-'*79))
    [void] $tpl.AppendLine('')
  }

  $fullNotices = Join-Path $OutDir 'THIRD-PARTY-LICENSES.txt'
  $tpl.ToString() | Set-Content -Encoding UTF8 $fullNotices
  Write-Host ("- Cargo-about full licenses: {0}" -f $fullNotices)
}
