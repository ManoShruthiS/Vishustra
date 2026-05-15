$taskName = "Ayan GhostBot"
$task = Get-ScheduledTask -TaskName $taskName
$settings = $task.Settings
$settings.DisallowStartIfOnBatteries = $false
$settings.StopIfGoingOnBatteries = $false
Set-ScheduledTask -TaskName $taskName -Settings $settings
Write-Host "SUCCESS: Task updated - will now run on battery power."
