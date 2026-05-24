param(
    [string[]]$PartPaths = @(
        "C:\Users\Admin\Downloads\20bn-something-something-v2-00",
        "C:\Users\Admin\Downloads\20bn-something-something-v2-01"
    ),
    [string]$LabelsZip = "C:\Users\Admin\Downloads\20bn-something-something-download-package-labels.zip",
    [string]$OutDir = "data\something_something_v2",
    [string]$Split = "train",
    [int]$MaxVideos = 0,
    [switch]$ExtractVideos,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Resolve-FullPath([string]$PathValue) {
    $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PathValue)
}

function Get-AvailableBytes([string]$PathValue) {
    $fullPath = Resolve-FullPath $PathValue
    $root = [System.IO.Path]::GetPathRoot($fullPath)
    ([System.IO.DriveInfo]::GetDrives() | Where-Object { $_.Name -eq $root }).AvailableFreeSpace
}

function Write-PartsToTar([string[]]$Parts, [string]$TarArguments) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "tar.exe"
    $psi.Arguments = $TarArguments
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    $process = [System.Diagnostics.Process]::Start($psi)
    $buffer = New-Object byte[] (4MB)
    try {
        foreach ($part in $Parts) {
            Write-Host "Streaming $part"
            $inputStream = [System.IO.File]::OpenRead($part)
            try {
                while (($read = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                    $process.StandardInput.BaseStream.Write($buffer, 0, $read)
                }
            }
            finally {
                $inputStream.Dispose()
            }
        }
        $process.StandardInput.Close()
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
    }
    finally {
        if (-not $process.HasExited) {
            $process.Kill()
        }
    }

    if ($stdout.Trim()) {
        Write-Host $stdout
    }
    if ($stderr.Trim()) {
        Write-Host $stderr
    }
    if ($process.ExitCode -ne 0) {
        throw "tar.exe failed with exit code $($process.ExitCode)"
    }
}

$outFull = Resolve-FullPath $OutDir
New-Item -ItemType Directory -Force -Path $outFull | Out-Null

foreach ($part in $PartPaths) {
    if (-not (Test-Path -LiteralPath $part)) {
        throw "Missing archive part: $part"
    }
}
if (-not (Test-Path -LiteralPath $LabelsZip)) {
    throw "Missing labels zip: $LabelsZip"
}

Write-Host "Extracting labels to $outFull"
tar.exe -xf $LabelsZip -C $outFull

if (-not $ExtractVideos) {
    Write-Host "Labels are ready. Re-run with -ExtractVideos to stream-extract videos."
    exit 0
}

$freeBytes = Get-AvailableBytes $outFull
$freeGb = [math]::Round($freeBytes / 1GB, 2)
if ($MaxVideos -eq 0 -and $freeBytes -lt 60GB -and -not $Force) {
    throw "Only $freeGb GB is free. Full extraction may not fit. Use -MaxVideos for a subset, free more disk, or pass -Force."
}

$tarArgs = "-xzf - -C `"$outFull`""
if ($MaxVideos -gt 0) {
    $splitPath = Join-Path $outFull "labels\$Split.json"
    if (-not (Test-Path -LiteralPath $splitPath)) {
        throw "Split file not found: $splitPath"
    }
    $rows = Get-Content -LiteralPath $splitPath -Raw | ConvertFrom-Json
    $selected = $rows | Select-Object -First $MaxVideos
    $includePath = Join-Path $outFull "something_v2_${Split}_${MaxVideos}_files.txt"
    $selected | ForEach-Object { "20bn-something-something-v2/$($_.id).webm" } | Set-Content -LiteralPath $includePath -Encoding ASCII
    $tarArgs = "$tarArgs -T `"$includePath`""
    Write-Host "Extracting $MaxVideos videos listed in $includePath"
}
else {
    Write-Host "Extracting all videos. This may take a while."
}

Write-PartsToTar -Parts $PartPaths -TarArguments $tarArgs
Write-Host "Something-Something V2 preparation complete: $outFull"
