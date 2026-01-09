# Ablation Study Script for ADKF-IFT
# Run 3 experiments with different configurations

Write-Host "========================================"
Write-Host "ADKF-IFT Ablation Experiments"
Write-Host "========================================"
Write-Host ""

$dataset = "./fs-mol-dataset"
$features = "gnn+ecfp+pc-descs+fc"

# Experiment configurations (3 experiments)
$experiments = @(
    @{
        Name = "Exp1: Adapter Only (--no-gating)"
        Steps = 50
        ValidateEvery = 50
        ExtraArgs = "--no-gating"
    },
    @{
        Name = "Exp2: Full Model (Gate + Adapter)"
        Steps = 50
        ValidateEvery = 50
        ExtraArgs = ""
    },
    @{
        Name = "Exp3: Gate Only 500 steps (--no-adapter)"
        Steps = 500
        ValidateEvery = 50
        ExtraArgs = "--no-adapter"
    }
)

# Record start time
$startTime = Get-Date
Write-Host "Start Time: $startTime"
Write-Host ""

# Run all experiments
$results = @()

foreach ($exp in $experiments) {
    Write-Host "========================================"
    Write-Host "Running: $($exp.Name)"
    Write-Host "========================================"
    
    $expStartTime = Get-Date
    
    # Build command
    $cmd = "python fs_mol/adaptive_dkt_train.py $dataset --features `"$features`" --num_train_steps $($exp.Steps) --validate_every $($exp.ValidateEvery)"
    if ($exp.ExtraArgs -ne "") {
        $cmd += " $($exp.ExtraArgs)"
    }
    
    Write-Host "Command: $cmd"
    Write-Host ""
    
    # Execute command
    Invoke-Expression $cmd
    
    $expEndTime = Get-Date
    $expDuration = $expEndTime - $expStartTime
    
    Write-Host ""
    Write-Host "Completed: $($exp.Name)"
    Write-Host "Duration: $($expDuration.ToString('hh\:mm\:ss'))"
    Write-Host ""
    
    $results += @{
        Name = $exp.Name
        Duration = $expDuration.ToString('hh\:mm\:ss')
    }
}

# Summary
$endTime = Get-Date
$totalDuration = $endTime - $startTime

Write-Host ""
Write-Host "========================================"
Write-Host "Experiment Summary"
Write-Host "========================================"
Write-Host "Start: $startTime"
Write-Host "End: $endTime"
Write-Host "Total Duration: $($totalDuration.ToString('hh\:mm\:ss'))"
Write-Host ""
Write-Host "Individual Durations:"
foreach ($r in $results) {
    Write-Host "  - $($r.Name): $($r.Duration)"
}
Write-Host ""
Write-Host "All experiments completed! Check outputs directory for results."
