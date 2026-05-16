$envFile = Get-Content ".env" -ErrorAction SilentlyContinue
$taskName = ($envFile | Select-String "TASK_NAME=").ToString().Split("=")[1].Trim()
if (-not $taskName) { $taskName = "System Maintenance Task" } # Fallback
$task = Get-ScheduledTask -TaskName $taskName
$settings = $task.Settings
$settings.DisallowStartIfOnBatteries = $false
$settings.StopIfGoingOnBatteries = $false
Set-ScheduledTask -TaskName $taskName -Settings $settings
Write-Host "SUCCESS: Task updated - will now run on battery power."
