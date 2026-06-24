# =====================================================================
# Registers the daily 08:00 LinkedIn (Israel) scrape with Windows Task
# Scheduler. Run once:   powershell -ExecutionPolicy Bypass -File .\register_scrape_task.ps1
# It runs even when the app/browser is closed, sharing the 800 budget.
# Remove with:   Unregister-ScheduledTask -TaskName "SmartCareer-DailyScrape" -Confirm:$false
# =====================================================================
$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Python = (Get-Command python).Source
$TaskName = "SmartCareer-DailyScrape"

$Action  = New-ScheduledTaskAction -Execute $Python -Argument "-m scraper.run_pipeline" -WorkingDirectory $ProjectDir
$Trigger = New-ScheduledTaskTrigger -Daily -At 8:00am
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "Daily 08:00 LinkedIn (Israel) scrape for Smart Career Navigator" -Force | Out-Null

Write-Host "Registered '$TaskName' -> daily at 08:00"
Write-Host "  Working dir: $ProjectDir"
Write-Host "  Command    : $Python -m scraper.run_pipeline"
