<#
.SYNOPSIS
  Set up a Windows machine to build, test, debug and format Zenith: MSYS2's CLANG64 toolchain, clang-format and the
  editor, and with -Training the NNUE trainer's Python environment.

.DESCRIPTION
  Each component is checked first and installed only when it is missing or not the version the repo wants, so the
  script is safe to run again at any time. Zenith builds on Windows in MSYS2's CLANG64 environment, as CI's Windows job
  does (README, "Windows"); MSYS2 goes to C:\msys64, a folder any user may create, and everything else into the user's
  own profile, so nothing needs an administrator. tools/setup-dev.sh is the same for Linux and macOS. VS Code's "setup
  environment" and "check environment" tasks run the one for the platform.

  What it sets up:
    - MSYS2 at C:\msys64, from the official base archive (its SHA-256 the digest GitHub publishes for the release
      asset), initialised and brought up to date with pacman -Syuu
    - in its CLANG64 environment: clang, lld, compiler-rt (the sanitizers of make debug), llvm-tools (llvm-cov and
      llvm-profdata), python (the Makefile's table and embedding scripts), gdb, and make, git and zip
    - uv, from Astral's release archive (its SHA-256 the one published beside it), into %USERPROFILE%\.local\bin
    - clang-format 22.1.8 through uv, the one version the pre-commit hook and CI's format gate accept, first on the
      PATH and, through MSYS2's ~/.bash_profile, ahead of the clang-format MSYS2's clang package carries
    - the repo's git hooks (core.hooksPath .githooks) and the VS Code extensions .vscode/extensions.json recommends
    - with -Training: a .venv in the repo with torch (CUDA 13.0 wheels) and numpy, for trainer/train.py

.PARAMETER Check
  Report what is present and what is missing, install nothing, and exit 1 if anything is missing.

.PARAMETER NoPersist
  Do not add anything to the user's PATH for good (only to this process's): for a test run in a scratch profile.
  MSYS2's own ~/.bash_profile, under its root, is still written.

.PARAMETER Training
  Also set up (or with -Check, check) the NNUE trainer's .venv: torch and numpy, and whether torch sees a CUDA GPU.

.PARAMETER Msys2Root
  Where MSYS2 is, or is installed to. C:\msys64 is what the README, the VS Code tasks and CI's Windows job use; another
  folder is for a test run that must leave the machine's own MSYS2 alone.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\setup-dev.ps1             # set up
  powershell -ExecutionPolicy Bypass -File tools\setup-dev.ps1 -Check      # report only
  powershell -ExecutionPolicy Bypass -File tools\setup-dev.ps1 -Training   # and the trainer's .venv
#>
param([switch]$Check, [switch]$NoPersist, [switch]$Training, [string]$Msys2Root = 'C:\msys64')
# Continue, not Stop: Windows PowerShell 5.1 makes a native command's stderr a terminating error under Stop, even
# redirected (pacman and uv print their progress there). The cmdlets that must stop say so; native commands' exit codes
# are checked.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
Set-Location (Split-Path -Parent $PSScriptRoot)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$clangFormatVersion = '22.1.8'   # CI's format gate (.github/workflows/ci.yml) and the pre-commit hook
$trainerPython = '3.13'
$torchIndex = 'https://download.pytorch.org/whl/cu130'
$msysPackages = @(
  'mingw-w64-clang-x86_64-clang', 'mingw-w64-clang-x86_64-lld', 'mingw-w64-clang-x86_64-compiler-rt',
  'mingw-w64-clang-x86_64-llvm-tools', 'mingw-w64-clang-x86_64-python', 'mingw-w64-clang-x86_64-gdb',
  'make', 'git', 'zip')

# ---------------------------------------------------------------------------------------------------------------------
# Helpers (the same in every reckless-technology repo's setup script).
$script:missing = 0
function Write-Ok([string]$name, [string]$detail) { Write-Host ("  ok       {0,-24} {1}" -f $name, $detail) -ForegroundColor Green }
function Write-Missing([string]$name, [string]$detail) {
  $script:missing++
  Write-Host ("  missing  {0,-24} {1}" -f $name, $detail) -ForegroundColor Yellow
}
function Write-Installing([string]$name, [string]$detail) { Write-Host ("  install  {0,-24} {1}" -f $name, $detail) -ForegroundColor Cyan }
function Write-Caution([string]$name, [string]$detail) { Write-Host ("  warning  {0,-24} {1}" -f $name, $detail) -ForegroundColor Magenta }

# Put a directory on this process's PATH and, unless -NoPersist, on the user's PATH for every new shell.
function Add-ToPath([string]$dir) {
  if (($env:PATH -split ';') -notcontains $dir) { $env:PATH = "$dir;$env:PATH" }
  if ($NoPersist -or $Check) { return }
  $user = [Environment]::GetEnvironmentVariable('Path', 'User')
  if ((($user -split ';') | Where-Object { $_ }) -notcontains $dir) {
    [Environment]::SetEnvironmentVariable('Path', (@($dir) + (($user -split ';') | Where-Object { $_ })) -join ';', 'User')
  }
}

# Download a file and refuse it unless its SHA-256 is the one given.
function Get-VerifiedFile([string]$url, [string]$sha256, [string]$out) {
  Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing -ErrorAction Stop
  $got = (Get-FileHash $out -Algorithm SHA256).Hash
  if ($got -ne $sha256.ToUpper()) { Remove-Item $out; throw "SHA-256 mismatch for $url (got $got, want $sha256)" }
}

# A downloaded checksum file's text, whatever type Invoke-WebRequest hands back.
function Get-Text([string]$url) {
  $content = (Invoke-WebRequest -Uri $url -UseBasicParsing -ErrorAction Stop).Content
  if ($content -is [byte[]]) { return [Text.Encoding]::ASCII.GetString($content) }
  return $content
}

function New-ScratchDir([string]$parent = [IO.Path]::GetTempPath()) {
  $dir = Join-Path $parent ("setup-dev-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
  New-Item -ItemType Directory -Path $dir -ErrorAction Stop | Out-Null
  return $dir
}

function Set-GitHooks([string]$git) {
  if (-not (Test-Path '.githooks') -or -not $git) { return }
  $current = & $git config core.hooksPath
  if ($current -eq '.githooks') { Write-Ok 'git hooks' 'core.hooksPath .githooks'; return }
  if ($Check) { Write-Missing 'git hooks' 'core.hooksPath is not .githooks'; return }
  & $git config core.hooksPath .githooks
  Write-Installing 'git hooks' 'core.hooksPath .githooks'
}

# The extensions .vscode/extensions.json recommends, when VS Code's command line is on the PATH.
function Install-VSCodeExtensions {
  $code = Get-Command code -ErrorAction SilentlyContinue
  if (-not $code -or -not (Test-Path '.vscode/extensions.json')) { return }
  $json = (Get-Content '.vscode/extensions.json' -Raw) -replace '(?m)^\s*//.*$', '' | ConvertFrom-Json
  $installed = & $code.Source --list-extensions 2>$null
  foreach ($ext in $json.recommendations) {
    if ($installed -contains $ext) { Write-Ok 'VS Code extension' $ext; continue }
    if ($Check) { Write-Missing 'VS Code extension' $ext; continue }
    Write-Installing 'VS Code extension' $ext
    & $code.Source --install-extension $ext --force | Out-Null
  }
}

# The version a clang-format reports ("clang-format version 22.1.8 (...)"), or '' when it cannot be run.
function Get-ClangFormatVersion([string]$exe) {
  $line = (& $exe --version 2>$null) | Select-Object -First 1
  if ($line -match 'version (\d+\.\d+\.\d+)') { return $Matches[1] }
  return ''
}

# uv's own directory: Astral's installer's rule, $XDG_BIN_HOME or else ~/.local/bin.
function Get-UvBinDir {
  if ($env:XDG_BIN_HOME) { return $env:XDG_BIN_HOME }
  return (Join-Path $env:USERPROFILE '.local\bin')
}

# ---------------------------------------------------------------------------------------------------------------------
# MSYS2.
$Msys2Root = [IO.Path]::GetFullPath($Msys2Root)
$msysBash = Join-Path $Msys2Root 'usr\bin\bash.exe'
$pacman = Join-Path $Msys2Root 'usr\bin\pacman.exe'

# A command in MSYS2's CLANG64 login shell, in the repo, its output streamed; returns its exit code.
function Invoke-Msys2([string]$command) {
  $env:MSYSTEM = 'CLANG64'
  $env:CHERE_INVOKING = '1'
  & $msysBash -lc $command | Out-Host
  return $LASTEXITCODE
}

# The whole system up to date. The first pass may update MSYS2's own runtime and core and stop there, so pacman's
# instructions are to run it again; a second pass with nothing to do is quick.
function Update-Msys2 {
  foreach ($pass in 1, 2) {
    $code = Invoke-Msys2 'pacman --noconfirm -Syuu'
    if ($code -ne 0) { throw "pacman -Syuu exited with $code" }
  }
}

# MSYS2's first login starts gpg-agent and dirmngr for the keyring and leaves them running; stop whatever runs from the
# root, so nothing holds its files once the script is done.
function Stop-Msys2Processes {
  Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and $_.Path.StartsWith($Msys2Root, [StringComparison]::OrdinalIgnoreCase) } |
    Stop-Process -Force -ErrorAction SilentlyContinue
}

# The base archive from the newest dated release of msys2/msys2-installer, refused unless its SHA-256 is the digest
# GitHub records for the asset, unpacked beside the root and moved into place, then initialised and updated.
function Install-Msys2 {
  $headers = @{ 'User-Agent' = 'zenith-setup-dev' }
  if ($env:GITHUB_TOKEN) { $headers['Authorization'] = "Bearer $env:GITHUB_TOKEN" }
  $releases = Invoke-RestMethod -Uri 'https://api.github.com/repos/msys2/msys2-installer/releases?per_page=10' -Headers $headers -ErrorAction Stop
  $asset = $null
  foreach ($release in $releases) {
    if ($release.draft -or $release.prerelease) { continue }
    $asset = $release.assets | Where-Object { $_.name -match '^msys2-base-x86_64-\d{8}\.sfx\.exe$' } | Select-Object -First 1
    if ($asset) { break }
  }
  if (-not $asset) { throw 'no msys2-base-x86_64-<date>.sfx.exe in the recent msys2-installer releases' }
  if ($asset.digest -notmatch '^sha256:([0-9a-f]{64})$') { throw "GitHub gives no SHA-256 digest for $($asset.name)" }
  $sha = $Matches[1]
  Write-Installing 'MSYS2' "$($asset.name) to $Msys2Root"
  $parent = Split-Path -Parent $Msys2Root
  if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -ErrorAction Stop | Out-Null }
  $scratch = New-ScratchDir
  $sfx = Join-Path $scratch $asset.name
  Get-VerifiedFile $asset.browser_download_url $sha $sfx
  # Unpacked on the root's own volume, so the move into place is a rename. The archive holds one folder, msys64.
  $unpack = New-ScratchDir $parent
  $p = Start-Process -FilePath $sfx -ArgumentList @('-y', "-o$unpack") -Wait -PassThru -NoNewWindow
  if ($p.ExitCode -ne 0) { throw "$($asset.name) exited with $($p.ExitCode)" }
  Move-Item -Path (Join-Path $unpack 'msys64') -Destination $Msys2Root -ErrorAction Stop
  Remove-Item -Recurse -Force $unpack, $scratch
  Write-Installing 'MSYS2' 'first login (the keyring), then pacman -Syuu'
  $code = Invoke-Msys2 ' '
  if ($code -ne 0) { throw "MSYS2's first login exited with $code" }
  Update-Msys2
  Stop-Msys2Processes
}

Write-Host "zenith: $(if ($Check) { 'checking' } else { 'setting up' }) the Windows development environment"

if (Test-Path $msysBash) { Write-Ok 'MSYS2' $Msys2Root }
elseif ($Check) { Write-Missing 'MSYS2' "$Msys2Root, and its CLANG64 toolchain" }
else { Install-Msys2 }

# The CLANG64 toolchain and the tools the Makefile and its targets call.
if (Test-Path $pacman) {
  $lacking = @($msysPackages | Where-Object { & $pacman -Q $_ 2>$null | Out-Null; $LASTEXITCODE -ne 0 })
  if ($lacking.Count -eq 0) { Write-Ok 'CLANG64 toolchain' ((& $pacman -Q mingw-w64-clang-x86_64-clang) -replace '^mingw-w64-clang-x86_64-', '') }
  elseif ($Check) { Write-Missing 'CLANG64 toolchain' ($lacking -join ' ') }
  else {
    # MSYS2 supports no partial upgrade: a package is installed onto a system brought up to date first.
    Write-Installing 'CLANG64 toolchain' ($lacking -join ' ')
    Update-Msys2
    $code = Invoke-Msys2 ("pacman --noconfirm -S --needed " + ($lacking -join ' '))
    if ($code -ne 0) { throw "pacman -S exited with $code" }
    Stop-Msys2Processes
  }
}

# uv, which installs clang-format (and, with -Training, the trainer's Python).
$uvBin = Get-UvBinDir
if (Test-Path (Join-Path $uvBin 'uv.exe')) { Add-ToPath $uvBin }
if (Get-Command uv -ErrorAction SilentlyContinue) { Write-Ok 'uv' (uv --version) }
elseif ($Check) { Write-Missing 'uv' "Astral's Python tool, which installs clang-format" }
else {
  # Astral's release archive, as its installer fetches it; the installer script for Windows checks no checksum, so the
  # archive is fetched here and held to the SHA-256 published beside it.
  $triple = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }
  Write-Installing 'uv' "uv-$triple.zip from Astral's latest release, to $uvBin"
  $scratch = New-ScratchDir
  $url = "https://github.com/astral-sh/uv/releases/latest/download/uv-$triple.zip"
  $sha = (Get-Text "$url.sha256").Trim().Split(' ')[0]
  Get-VerifiedFile $url $sha (Join-Path $scratch 'uv.zip')
  Expand-Archive -Path (Join-Path $scratch 'uv.zip') -DestinationPath (Join-Path $scratch 'uv') -ErrorAction Stop
  New-Item -ItemType Directory -Path $uvBin -Force -ErrorAction Stop | Out-Null
  Get-ChildItem (Join-Path $scratch 'uv') -Recurse -Filter '*.exe' | Copy-Item -Destination $uvBin -Force -ErrorAction Stop
  Remove-Item -Recurse -Force $scratch
  Add-ToPath $uvBin
}

# clang-format 22.1.8, the first on the PATH: what the pre-commit hook (run by Git for Windows' bash) and make format
# find. A different version restyles code that is already clean, and the hook then skips its check.
$cf = Get-Command clang-format -ErrorAction SilentlyContinue | Select-Object -First 1
$cfVersion = if ($cf) { Get-ClangFormatVersion $cf.Source } else { '' }
if ($cfVersion -eq $clangFormatVersion) { Write-Ok 'clang-format' "$cfVersion ($($cf.Source))" }
elseif ($Check) { Write-Missing 'clang-format' "$clangFormatVersion (first on the PATH: $(if ($cf) { "$cfVersion, $($cf.Source)" } else { 'none' }))" }
elseif (Get-Command uv -ErrorAction SilentlyContinue) {
  Write-Installing 'clang-format' "$clangFormatVersion through uv tool install"
  uv tool install --force "clang-format==$clangFormatVersion"
  if ($LASTEXITCODE -ne 0) { throw "uv tool install exited with $LASTEXITCODE" }
  $toolBin = (uv tool dir --bin).Trim()
  Add-ToPath $toolBin
  $cf = Get-Command clang-format -ErrorAction SilentlyContinue | Select-Object -First 1
  $cfVersion = if ($cf) { Get-ClangFormatVersion $cf.Source } else { '' }
  if ($cfVersion -ne $clangFormatVersion) {
    Write-Caution 'clang-format' "$($cf.Source) ($cfVersion) shadows uv's in ${toolBin}: remove it, or put $toolBin ahead of it"
  }
}

# A CLANG64 shell builds its own PATH, which leaves the user's out and puts /clang64/bin first, and MSYS2's clang
# package carries a clang-format of its own version there: the one make format, the VS Code tasks and a commit made with
# MSYS2's git would find. The directory of the 22.1.8 found above goes ahead of it in MSYS2's ~/.bash_profile, which a
# login shell reads after /etc/profile has set its PATH - never a directory that holds a clang of its own, which would
# take the place of CLANG64's.
function Get-Msys2ClangFormat {
  $env:MSYSTEM = 'CLANG64'
  $lines = @(& $msysBash -lc 'command -v clang-format && clang-format --version' 2>$null)
  if ($lines.Count -ge 2 -and $lines[1] -match 'version (\d+\.\d+\.\d+)') { return @($lines[0], $Matches[1]) }
  return @('none', '')
}
if (Test-Path $msysBash) {
  $msysPath, $msysVersion = Get-Msys2ClangFormat
  if ($msysVersion -eq $clangFormatVersion) { Write-Ok 'CLANG64 clang-format' "$msysVersion ($msysPath)" }
  elseif ($Check) { Write-Missing 'CLANG64 clang-format' "$clangFormatVersion (first on a CLANG64 shell's PATH: $msysPath $msysVersion)" }
  elseif ($cfVersion -ne $clangFormatVersion) { Write-Caution 'CLANG64 clang-format' "$msysPath is $msysVersion, and no $clangFormatVersion to put ahead of it" }
  else {
    $cfDir = Split-Path -Parent $cf.Source
    if (Test-Path (Join-Path $cfDir 'clang.exe')) {
      Write-Caution 'CLANG64 clang-format' "$msysPath is $msysVersion; $cfDir holds a clang too, so it stays off a CLANG64 shell's PATH"
    }
    else {
      $posixDir = '/' + $cfDir.Substring(0, 1).ToLower() + ($cfDir.Substring(2) -replace '\\', '/')
      $line = "export PATH=`"${posixDir}:`$PATH`""
      $bashProfile = ((& $msysBash -lc 'cygpath -w ~/.bash_profile') | Select-Object -First 1).Trim()
      $present = (Test-Path $bashProfile) -and ((Get-Content $bashProfile) -contains $line)
      if (-not $present) {
        Write-Installing 'CLANG64 clang-format' "$cfDir ahead of $msysPath, in $bashProfile"
        # LF line endings: MSYS2's bash reads a carriage return as part of the line.
        [IO.File]::AppendAllText($bashProfile, "`n# clang-format $clangFormatVersion ahead of MSYS2's own (tools/setup-dev.ps1)`n$line`n")
      }
      $msysPath, $msysVersion = Get-Msys2ClangFormat
      if ($msysVersion -ne $clangFormatVersion) { Write-Caution 'CLANG64 clang-format' "$msysPath is $msysVersion still" }
      elseif ($present) { Write-Ok 'CLANG64 clang-format' "$msysVersion ($msysPath)" }
    }
  }
}

# The git hooks, through Git for Windows' git where there is one, else MSYS2's.
$gitExe = (Get-Command git -ErrorAction SilentlyContinue | Select-Object -First 1).Source
if (-not $gitExe -and (Test-Path (Join-Path $Msys2Root 'usr\bin\git.exe'))) { $gitExe = Join-Path $Msys2Root 'usr\bin\git.exe' }
Set-GitHooks $gitExe
Install-VSCodeExtensions

# The NNUE trainer's .venv: torch from PyTorch's CUDA 13.0 index (with PyPI for everything that index lacks) and numpy,
# at trainer/requirements.txt's floors.
if ($Training) {
  $venvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
  $probe = "import torch, numpy; c = torch.cuda.is_available(); print(torch.__version__, numpy.__version__, int(c), torch.cuda.get_device_name(0) if c else '', sep='|')"
  $state = if (Test-Path $venvPython) { (& $venvPython -c $probe 2>$null) | Select-Object -Last 1 } else { '' }
  if (-not $state -and -not $Check) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv is needed for the trainer .venv' }
    if (-not (Test-Path $venvPython)) {
      Write-Installing 'trainer .venv' "Python $trainerPython through uv"
      uv venv .venv --python $trainerPython
      if ($LASTEXITCODE -ne 0) { throw "uv venv exited with $LASTEXITCODE" }
    }
    Write-Installing 'trainer .venv' "torch (CUDA 13.0) and numpy, from trainer/requirements.txt"
    uv pip install --python $venvPython -r trainer/requirements.txt --index-url $torchIndex --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) { throw "uv pip install exited with $LASTEXITCODE" }
    $state = (& $venvPython -c $probe 2>$null) | Select-Object -Last 1
  }
  if (-not $state) { Write-Missing 'trainer .venv' '.venv with torch and numpy (tools\setup-dev.ps1 -Training)' }
  else {
    $torch, $numpy, $cuda, $device = $state.Split('|')
    if ($cuda -eq '1') { Write-Ok 'trainer .venv' "torch $torch, numpy $numpy, CUDA on $device" }
    else { Write-Caution 'trainer .venv' "torch $torch, numpy $numpy, but no CUDA GPU: training runs on the CPU" }
  }
}

if ($Check) {
  if ($script:missing) { Write-Host "$($script:missing) missing: run tools\setup-dev.ps1 (VS Code: the 'setup environment' task)."; exit 1 }
  Write-Host 'Everything is in place.'
}
else {
  Write-Host 'Done. Open a new terminal (or restart VS Code) for PATH changes to reach it, then, in a CLANG64 shell'
  Write-Host "($Msys2Root\clang64.exe) or through VS Code's make tasks: make check"
}
