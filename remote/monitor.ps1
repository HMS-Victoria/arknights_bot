# AutoDL training progress monitor for PowerShell.
#
# Reads the remote training log over SSH and renders a live progress panel:
# step/total, ETA, loss, and the tail of the log.
#
# Usage:
#   .\remote\monitor.ps1                                      # refresh every 10 seconds
#   .\remote\monitor.ps1 -SshTarget root@connect.example.com -Port 12345
#   .\remote\monitor.ps1 -RefreshSeconds 5
#   .\remote\monitor.ps1 -Once                                # single refresh, then exit
#   .\remote\monitor.ps1 -SelfTest                            # offline parser check, no SSH
#
# Credentials are never hardcoded here: pass -Key, or define a Host alias in
# ~/.ssh/config and pass -SshTarget <alias>.

[CmdletBinding()]
param(
    [string]$Key = "",
    [string]$SshTarget = "",
    [int]$Port = 22,
    [int]$RefreshSeconds = 10,
    [string]$TrainLogPath = "/root/train.log",
    [string]$AutostartLogPath = "/root/autostart.log",
    [string]$PidPath = "/root/train.pid",
    [switch]$Once,
    [switch]$SelfTest
)

$ErrorActionPreference = "Continue"

function ConvertFrom-AnsiText {
    param([string]$Text)
    if ([string]::IsNullOrEmpty($Text)) { return "" }
    $esc = [string][char]27
    return ($Text -replace "`r", "") -replace "$esc\[[0-9;?]*[A-Za-z]", ""
}

function ConvertTo-Seconds {
    param([string]$TimeText)
    if ([string]::IsNullOrWhiteSpace($TimeText)) { return $null }
    $parts = @($TimeText.Trim() -split ":")
    $total = 0.0
    foreach ($part in $parts) {
        $total = $total * 60 + [double]$part
    }
    return $total
}

function Format-Duration {
    param([object]$Seconds)
    if ($null -eq $Seconds) { return "--" }
    $value = [double]$Seconds
    if ([double]::IsNaN($value) -or $value -lt 0) { return "--" }
    $ts = [TimeSpan]::FromSeconds($value)
    if ($ts.TotalHours -ge 1) {
        return "{0}h {1:D2}m {2:D2}s" -f [math]::Floor($ts.TotalHours), $ts.Minutes, $ts.Seconds
    }
    return "{0:D2}m {1:D2}s" -f $ts.Minutes, $ts.Seconds
}

function Format-Bar {
    param([object]$Percent, [int]$Width = 44)
    if ($null -eq $Percent) { return (" " * $Width) }
    $value = [double]$Percent
    if ([double]::IsNaN($value)) { return (" " * $Width) }
    $clamped = [math]::Max(0.0, [math]::Min(100.0, $value))
    $filled = [math]::Floor($Width * $clamped / 100.0)
    return ("#" * [int]$filled) + ("-" * ($Width - [int]$filled))
}

function Parse-TrainLog {
    param([string]$LogText)

    $result = [pscustomobject][ordered]@{
        State = "unknown"
        CurrentStep = $null
        TotalSteps = $null
        Percent = $null
        SecondsPerStep = $null
        ElapsedSeconds = $null
        RemainingSeconds = $null
        TrainLoss = $null
        EvalLoss = $null
        LearningRate = $null
        Epoch = $null
        DatasetSize = $null
        EffectiveBatch = $null
        Epochs = $null
        LastError = ""
        RecentLines = @()
    }

    if ([string]::IsNullOrWhiteSpace($LogText)) { return $result }

    $recent = New-Object System.Collections.ArrayList
    $progressPattern = '(\d+)\s*/\s*(\d+)\s+\[((?:\d+:)?\d+:\d+)<((?:\d+:)?\d+:\d+),\s*([0-9.]+)(s/it|it/s)\]'
    $progressNoRatePattern = '(\d+)\s*/\s*(\d+)\s+\[((?:\d+:)?\d+:\d+)<((?:\d+:)?\d+:\d+)\]'
    $floatPattern = '-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'

    foreach ($line in ($LogText -split "`n")) {
        $clean = (ConvertFrom-AnsiText -Text $line).Trim()
        if (-not $clean) { continue }

        if ($clean -match $progressPattern) {
            $result.CurrentStep = [int]$Matches[1]
            $result.TotalSteps = [int]$Matches[2]
            $result.ElapsedSeconds = ConvertTo-Seconds $Matches[3]
            $result.RemainingSeconds = ConvertTo-Seconds $Matches[4]
            $rate = [double]$Matches[5]
            if ($Matches[6] -eq "it/s") {
                $result.SecondsPerStep = if ($rate -gt 0) { 1.0 / $rate } else { $null }
            } else {
                $result.SecondsPerStep = $rate
            }
        } elseif ($clean -match $progressNoRatePattern) {
            $result.CurrentStep = [int]$Matches[1]
            $result.TotalSteps = [int]$Matches[2]
            $result.ElapsedSeconds = ConvertTo-Seconds $Matches[3]
            $result.RemainingSeconds = ConvertTo-Seconds $Matches[4]
        }

        if ($clean -match "dataset=(\d+)\s+effective_batch=(\d+)\s+epochs=([0-9.]+)") {
            $result.DatasetSize = [int]$Matches[1]
            $result.EffectiveBatch = [int]$Matches[2]
            $result.Epochs = [double]$Matches[3]
        }
        if ($clean -match "'loss':\s*($floatPattern)") { $result.TrainLoss = [double]$Matches[1] }
        if ($clean -match "'eval_loss':\s*($floatPattern)") { $result.EvalLoss = [double]$Matches[1] }
        if ($clean -match "'learning_rate':\s*($floatPattern)") { $result.LearningRate = [double]$Matches[1] }
        if ($clean -match "'epoch':\s*($floatPattern)") { $result.Epoch = [double]$Matches[1] }
        if ($clean -match "Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError|KeyboardInterrupt") {
            $result.State = "error"
            $result.LastError = $clean
        } elseif ($clean -match "training finished") {
            $result.State = "finished"
        } elseif ($clean -match "Running training") {
            if ($result.State -ne "error" -and $result.State -ne "finished") {
                $result.State = "training"
            }
        }

        $isProgressLine = $clean -match '\d+\s*/\s*\d+\s+\['
        if (-not $isProgressLine) {
            $null = $recent.Add($clean)
            while ($recent.Count -gt 4) { $recent.RemoveAt(0) }
        }
    }

    if ($null -eq $result.TotalSteps -and $result.DatasetSize -gt 0 -and $result.EffectiveBatch -gt 0 -and $null -ne $result.Epochs) {
        $result.TotalSteps = [math]::Ceiling($result.DatasetSize / $result.EffectiveBatch * $result.Epochs)
    }
    if ($result.TotalSteps -gt 0 -and $null -ne $result.CurrentStep) {
        $result.Percent = [math]::Min(100.0, [math]::Round(100.0 * $result.CurrentStep / $result.TotalSteps, 1))
    }

    $result.RecentLines = @($recent)
    return $result
}

function Get-Section {
    param([string]$Text, [string]$Start, [string]$End)
    if ([string]::IsNullOrEmpty($Text)) { return "" }
    $startIndex = $Text.IndexOf($Start)
    if ($startIndex -lt 0) { return "" }
    $startIndex += $Start.Length
    $endIndex = -1
    if ($End) { $endIndex = $Text.IndexOf($End, $startIndex) }
    if ($endIndex -lt 0) { $endIndex = $Text.Length }
    return $Text.Substring($startIndex, $endIndex - $startIndex).Trim()
}

function Get-RemoteStatus {
    param(
        [string]$Key,
        [string]$SshTarget,
        [int]$Port,
        [string]$TrainLogPath,
        [string]$AutostartLogPath,
        [string]$PidPath
    )

    $remoteCommand = "echo '===PROC==='; ps -eo pid,args | grep -E '[p]ython .*train_lora\.py|[b]ash\s+[^ ]*run_train\.sh|[p]ip .*install' | grep -v 'bash -c' || true; echo '===AUTOSTART==='; tail -n 60 $AutostartLogPath 2>/dev/null; echo '===TRAIN==='; tail -n 600 $TrainLogPath 2>/dev/null; echo '===PID==='; cat $PidPath 2>/dev/null"
    $sshArgs = @("-p", $Port, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15")
    if ($Key) { $sshArgs += @("-i", $Key) }
    $sshArgs += @($SshTarget, $remoteCommand)
    $rawLines = & ssh @sshArgs 2>&1
    $exitCode = $LASTEXITCODE
    $raw = $rawLines -join "`n"

    if ($exitCode -ne 0) {
        return [pscustomobject]@{
            Process = ""
            Autostart = ""
            Train = ""
            Error = $raw
        }
    }

    return [pscustomobject]@{
        Process = Get-Section -Text $raw -Start "===PROC===" -End "===AUTOSTART==="
        Autostart = Get-Section -Text $raw -Start "===AUTOSTART===" -End "===TRAIN==="
        Train = Get-Section -Text $raw -Start "===TRAIN===" -End "===PID==="
        Error = ""
    }
}

function Get-OverallState {
    param($Parsed, [string]$ProcessText, [string]$AutostartText, [string]$RemoteError)
    if ($RemoteError) { return "ssh-error" }
    if ($Parsed.State -eq "error") { return "error" }
    if ($Parsed.State -eq "finished") { return "finished" }
    if ($null -ne $Parsed.CurrentStep -and $Parsed.TotalSteps -gt 0) { return "training" }
    if ($Parsed.State -eq "training") { return "training" }
    if ($ProcessText -match "pip install") { return "setup" }
    if ($ProcessText -match "train_lora|run_train.sh") { return "loading" }
    if ($AutostartText -match "starting training") { return "waiting" }
    return "idle"
}

function Write-TrainingStatus {
    param(
        $Parsed,
        [string]$State,
        [string]$ProcessText,
        [string]$AutostartText,
        [string]$RemoteError,
        [string]$Timestamp
    )

    $colors = @{
        "ssh-error" = "Red"
        "error" = "Red"
        "finished" = "Green"
        "training" = "Green"
        "loading" = "Cyan"
        "setup" = "Yellow"
        "waiting" = "Yellow"
        "idle" = "Gray"
    }
    $color = "White"
    if ($colors.ContainsKey($State)) { $color = $colors[$State] }

    $statusText = $State.ToUpper()
    if ($State -eq "loading") { $statusText = "LOADING MODEL / DATASET" }

    Write-Host ""
    Write-Host ("===== AutoDL training progress | {0} =====" -f $Timestamp) -ForegroundColor Cyan
    Write-Host ""
    Write-Host ("STATUS   : {0}" -f $statusText) -ForegroundColor $color

    if ($null -ne $Parsed.CurrentStep -and $null -ne $Parsed.TotalSteps) {
        $bar = Format-Bar -Percent $Parsed.Percent -Width 44
        Write-Host ("STEP     : {0} / {1}" -f $Parsed.CurrentStep, $Parsed.TotalSteps) -ForegroundColor Cyan
        Write-Host ("BAR      : [{0}] {1}%" -f $bar, $Parsed.Percent)
    } else {
        Write-Host "STEP     : --"
    }

    $speedText = "--"
    if ($null -ne $Parsed.SecondsPerStep) {
        $speedText = "{0:F2} s/it" -f $Parsed.SecondsPerStep
    }
    $etaText = "--"
    if ($null -ne $Parsed.RemainingSeconds) {
        $etaText = Format-Duration $Parsed.RemainingSeconds
        if ($Parsed.RemainingSeconds -gt 0) {
            $etaClock = (Get-Date).AddSeconds($Parsed.RemainingSeconds).ToString("HH:mm:ss")
            $etaText += " (finish ~$etaClock)"
        }
    }

    Write-Host ("SPEED    : {0}" -f $speedText)
    Write-Host ("ELAPSED  : {0}" -f (Format-Duration $Parsed.ElapsedSeconds))
    Write-Host ("ETA      : {0}" -f $etaText)

    $lossParts = @()
    if ($null -ne $Parsed.TrainLoss) { $lossParts += "train={0:F4}" -f $Parsed.TrainLoss }
    if ($null -ne $Parsed.EvalLoss) { $lossParts += "eval={0:F4}" -f $Parsed.EvalLoss }
    if ($null -ne $Parsed.LearningRate) { $lossParts += "lr={0:E2}" -f $Parsed.LearningRate }
    if ($null -ne $Parsed.Epoch) { $lossParts += "epoch={0:F2}" -f $Parsed.Epoch }
    if ($lossParts.Count -gt 0) {
        Write-Host ("LOSS     : {0}" -f ($lossParts -join "  "))
    }

    if ($ProcessText) {
        $firstProc = ($ProcessText -split "`n" | Select-Object -First 1)
        if ($firstProc) {
            Write-Host ("PROCESS  : {0}" -f $firstProc.Trim()) -ForegroundColor DarkGray
        }
    }
    if ($AutostartText -and ($State -eq "setup" -or $State -eq "waiting")) {
        $lastSetup = ($AutostartText -split "`n" | Select-Object -Last 1)
        if ($lastSetup) {
            Write-Host ("SETUP    : {0}" -f $lastSetup.Trim()) -ForegroundColor Yellow
        }
    }
    if ($RemoteError) {
        Write-Host ("SSH ERROR: {0}" -f ($RemoteError -replace "`n", " ")) -ForegroundColor Red
    }
    if ($Parsed.LastError) {
        Write-Host ("ERROR    : {0}" -f $Parsed.LastError) -ForegroundColor Red
    }

    if ($Parsed.RecentLines.Count -gt 0) {
        Write-Host ""
        Write-Host "RECENT LOG:"
        foreach ($line in $Parsed.RecentLines) {
            Write-Host ("  " + $line) -ForegroundColor DarkGray
        }
    }

    Write-Host ""
    Write-Host "Press Ctrl+C to stop monitoring." -ForegroundColor DarkGray

    if ($null -ne $Parsed.Percent) {
        Write-Progress -Activity "AutoDL training" -Status ("Step {0}/{1} ({2}%)" -f $Parsed.CurrentStep, $Parsed.TotalSteps, $Parsed.Percent) -PercentComplete $Parsed.Percent -CurrentOperation ("ETA " + (Format-Duration $Parsed.RemainingSeconds))
    } elseif ($null -ne $Parsed.CurrentStep -and $null -ne $Parsed.TotalSteps) {
        Write-Progress -Activity "AutoDL training" -Status ("Step {0}/{1}" -f $Parsed.CurrentStep, $Parsed.TotalSteps) -PercentComplete 0
    } else {
        Write-Progress -Activity "AutoDL training" -Status $State -Completed
    }
}

if ($SelfTest) {
    $sample = @'
  0%|          | 0/522 [00:00<?, ?it/s]
 10/522 [00:12<10:15, 1.20s/it]
 20/522 [00:24<10:02, 1.21s/it] {'loss': 1.2345, 'grad_norm': 2.3, 'learning_rate': 0.0002, 'epoch': 0.02}
 30/522 [00:36<09:50, 1.20s/it] {'eval_loss': 1.752, 'eval_runtime': 3.1, 'eval_samples_per_second': 30.0, 'eval_steps': 30}
training finished, wall time: 10.5 minutes
'@
    $parsed = Parse-TrainLog -LogText $sample
    Write-TrainingStatus -Parsed $parsed -State "finished" -ProcessText "python train/train_lora.py --epochs 3" -AutostartText "starting training" -RemoteError "" -Timestamp (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    exit 0
}

while ($true) {
    Clear-Host
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $remote = Get-RemoteStatus -Key $Key -SshTarget $SshTarget -Port $Port -TrainLogPath $TrainLogPath -AutostartLogPath $AutostartLogPath -PidPath $PidPath
    $parsed = Parse-TrainLog -LogText $remote.Train
    $state = Get-OverallState -Parsed $parsed -ProcessText $remote.Process -AutostartText $remote.Autostart -RemoteError $remote.Error
    Write-TrainingStatus -Parsed $parsed -State $state -ProcessText $remote.Process -AutostartText $remote.Autostart -RemoteError $remote.Error -Timestamp $timestamp

    if ($Once) { break }
    Start-Sleep -Seconds $RefreshSeconds
}
