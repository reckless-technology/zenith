# What a Windows build of the engine loads from the system: the DLLs its import table and its delay-load table name,
# held to an allowlist of Windows' own. CI's Windows job and the release's Windows build run it on the engines they
# built - zenith.exe and zenith-pext.exe - and fail on anything else.
#
# Windows loads every DLL an executable imports before main runs, and an executable whose import the machine cannot
# satisfy does not start at all (0xC0000135) - a GUI that launches the engine sees it die with no output. The engines
# link MSYS2 CLANG64's runtime statically (-static, RELEASE_LDFLAGS in the Makefile), so they run on a Windows with no
# MSYS2 or MinGW installed: no libwinpthread-1.dll, libgcc, libc++ or libunwind may appear, nor a Visual C++ runtime,
# and one that does - a build that lost the flag - is named as a runtime DLL. The C runtime is the UCRT, which CLANG64
# links through its api-ms-win-crt-*.dll API sets: those are Windows' own from Windows 10 on, and allowed.
#
# The tables are read from the PE file itself - the import directory's descriptors and the delay-load directory's, each
# naming its DLL by an RVA mapped to the file through the section table - so it needs no dumpbin or llvm-objdump, and
# runs on any runner, in Windows PowerShell 5.1 or PowerShell 7. It lists what `dumpbin /dependents` lists.
#
# Usage: pwsh -NoProfile -File tools/windows-dependents.ps1 <binary>... [-ReportOnly]
#   -ReportOnly  a DLL outside the allowlist is a warning and the exit is 0
param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)] [string[]] $Binaries,
    [switch] $ReportOnly
)
$ErrorActionPreference = 'Stop'

# The DLLs the engine may import: Windows' own, on every Windows it runs on. The kernel - files, threads, the clock,
# the console, the executable's own path - and the UCRT's API sets (api-ms-win-crt-*), the C runtime itself.
$allowed = @('KERNEL32.dll')
$allowedPattern = '^api-ms-win-crt-[a-z]+-l1-1-0\.dll$'
# The toolchains' runtime DLLs, which a static link keeps out: named for what they are when one appears.
$runtime = '^(libwinpthread|libgcc|libstdc\+\+|libc\+\+|libunwind|libssp|libgomp|vcruntime|msvcp|msvcr|concrt|vcomp)'

# The DLLs a PE file imports and delay-loads, as two lists, read from its import and delay-load directories.
function Get-Imports([string] $file) {
    $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $file).Path)
    if ($bytes.Length -lt 64 -or [BitConverter]::ToUInt16($bytes, 0) -ne 0x5A4D) { throw "${file}: not a PE file" }
    $pe = [BitConverter]::ToInt32($bytes, 0x3C)
    if ([BitConverter]::ToUInt32($bytes, $pe) -ne 0x4550) { throw "${file}: no PE signature" }
    $coff = $pe + 4
    $sectionCount = [BitConverter]::ToUInt16($bytes, $coff + 2)
    $optionalSize = [BitConverter]::ToUInt16($bytes, $coff + 16)
    $optional = $coff + 20
    $magic = [BitConverter]::ToUInt16($bytes, $optional)
    # The data directories follow the optional header's fixed part: 96 bytes in PE32, 112 in PE32+.
    $directories = $optional + $(if ($magic -eq 0x20B) { 112 } elseif ($magic -eq 0x10B) { 96 } else {
            throw "${file}: optional header magic $magic" })
    $sections = $optional + $optionalSize
    # A relative virtual address as an offset into the file, through the section that holds it.
    $toOffset = {
        param([uint32] $rva)
        for ($i = 0; $i -lt $sectionCount; $i++) {
            $header = $sections + 40 * $i
            $virtualSize = [BitConverter]::ToUInt32($bytes, $header + 8)
            $virtualAddress = [BitConverter]::ToUInt32($bytes, $header + 12)
            $rawSize = [BitConverter]::ToUInt32($bytes, $header + 16)
            $rawPointer = [BitConverter]::ToUInt32($bytes, $header + 20)
            $extent = [Math]::Max($virtualSize, $rawSize)
            if ($rva -ge $virtualAddress -and $rva -lt $virtualAddress + $extent) {
                return [int]($rva - $virtualAddress + $rawPointer)
            }
        }
        throw "${file}: RVA $rva is in no section"
    }
    $nameAt = {
        param([uint32] $rva)
        $start = & $toOffset $rva
        $end = $start
        while ($bytes[$end] -ne 0) { $end++ }
        return [System.Text.Encoding]::ASCII.GetString($bytes, $start, $end - $start)
    }
    # Each directory's descriptors, ended by an all-zero one: an import descriptor is 20 bytes with its DLL's name at
    # +12, a delay-load descriptor 32 bytes with it at +4.
    $read = {
        param([int] $index, [int] $size, [int] $nameField)
        $names = @()
        $rva = [BitConverter]::ToUInt32($bytes, $directories + 8 * $index)
        if ($rva -eq 0) { return , $names }
        $at = & $toOffset $rva
        while ($true) {
            $nameRva = [BitConverter]::ToUInt32($bytes, $at + $nameField)
            if ($nameRva -eq 0) { break }
            $names += & $nameAt $nameRva
            $at += $size
        }
        return , $names
    }
    return , @((& $read 1 20 12), (& $read 13 32 4))
}

$outside = @()
foreach ($file in $Binaries) {
    if (-not (Test-Path -LiteralPath $file)) { throw "$file is not there - build it first" }
    $label = Split-Path -Leaf $file
    $imports = Get-Imports $file
    $dlls = @($imports[0])
    $delayed = @($imports[1])
    if ($dlls.Count -eq 0) { throw "${label}: no import read - is it a Windows executable?" }
    $text = ($dlls | Sort-Object) -join ' '
    if ($delayed.Count -gt 0) { $text += "; delay-loaded: $(($delayed | Sort-Object) -join ' ')" }
    Write-Host "${label}: $text"
    foreach ($dll in $dlls + $delayed) {
        if ($dll -match $runtime) {
            $outside += "${label}: $dll, a toolchain runtime DLL - the engines link it statically (-static)"
        } elseif (-not ($allowed | Where-Object { $_ -ieq $dll }) -and $dll -notmatch $allowedPattern) {
            $outside += "${label}: $dll, a DLL outside the allowlist (tools/windows-dependents.ps1)"
        }
    }
}
if ($outside.Count -eq 0) {
    Write-Host "$($Binaries.Count) binaries import system DLLs on the allowlist alone, and no toolchain runtime DLL"
    exit 0
}
$level = if ($ReportOnly) { 'warning' } else { 'error' }
foreach ($line in $outside) { Write-Host "::${level}::$line" }
if ($ReportOnly) { exit 0 }
exit 1
