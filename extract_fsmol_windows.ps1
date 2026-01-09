# Windows FSMOL Dataset Extraction Script
# Extract fsmol.tar file for ADKF-IFT project

Write-Host "======================================"
Write-Host " ADKF-IFT Dataset Extraction Tool"
Write-Host "======================================"
Write-Host ""

# Define paths
$TarFile = ".\fsmol.tar"
$OutputDir = ".\fs-mol-dataset"

# Check if tar file exists
if (-Not (Test-Path $TarFile)) {
    Write-Host "[ERROR] fsmol.tar file not found!" -ForegroundColor Red
    Write-Host "Please make sure fsmol.tar is in the current directory." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Current directory: $(Get-Location)"
    Pause
    Exit 1
}

Write-Host "[OK] Found tar file: $TarFile" -ForegroundColor Green
$FileSize = (Get-Item $TarFile).Length / 1MB
Write-Host "    File size: $([Math]::Round($FileSize, 2)) MB" -ForegroundColor Gray
Write-Host ""

# Check if output directory already exists
if (Test-Path $OutputDir) {
    Write-Host "[WARNING] Directory $OutputDir already exists!" -ForegroundColor Yellow
    $Overwrite = Read-Host "Delete and re-extract? (y/N)"
    if ($Overwrite -eq "y" -or $Overwrite -eq "Y") {
        Write-Host "Deleting old directory..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $OutputDir
    } else {
        Write-Host "Skipping extraction." -ForegroundColor Yellow
        Pause
        Exit 0
    }
}

Write-Host "======================================"
Write-Host " Select Extraction Method"
Write-Host "======================================"
Write-Host ""
Write-Host "1. Use WSL (Recommended if installed)"
Write-Host "2. Use tar command (Windows 10 1803+)"
Write-Host "3. Use 7-Zip (Requires 7-Zip installed)"
Write-Host ""

$Choice = Read-Host "Please select (1/2/3)"

switch ($Choice) {
    "1" {
        # Use WSL
        Write-Host ""
        Write-Host "[Method 1] Using WSL to extract..." -ForegroundColor Cyan
        Write-Host ""
        
        # Check if WSL is available
        try {
            $WslCheck = wsl --status 2>&1
            Write-Host "[OK] WSL is available" -ForegroundColor Green
            
            # Convert path to WSL format
            $CurrentDir = (Get-Location).Path
            $WslPath = $CurrentDir -replace '\\', '/' -replace 'C:', '/mnt/c'
            
            Write-Host "Extracting (this may take a few minutes)..." -ForegroundColor Yellow
            Write-Host ""
            
            # Execute tar command using WSL
            wsl bash -c "cd '$WslPath' && tar -xf fsmol.tar && mv fs-mol fs-mol-dataset"
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host ""
                Write-Host "[OK] Extraction successful!" -ForegroundColor Green
            } else {
                Write-Host ""
                Write-Host "[ERROR] Extraction failed! Exit code: $LASTEXITCODE" -ForegroundColor Red
                Pause
                Exit 1
            }
        } catch {
            Write-Host "[ERROR] WSL not available!" -ForegroundColor Red
            Write-Host "Error: $_" -ForegroundColor Red
            Write-Host ""
            Write-Host "Please install WSL or choose another method." -ForegroundColor Yellow
            Pause
            Exit 1
        }
    }
    
    "2" {
        # Use Windows tar
        Write-Host ""
        Write-Host "[Method 2] Using Windows tar command..." -ForegroundColor Cyan
        Write-Host ""
        
        # Check if tar command is available
        try {
            $TarVersion = tar --version 2>&1
            Write-Host "[OK] tar command is available" -ForegroundColor Green
            
            Write-Host "Extracting (this may take a few minutes)..." -ForegroundColor Yellow
            Write-Host ""
            
            # Extract
            tar -xf $TarFile
            
            if ($LASTEXITCODE -eq 0) {
                # Rename directory
                if (Test-Path ".\fs-mol") {
                    Rename-Item -Path ".\fs-mol" -NewName "fs-mol-dataset"
                }
                Write-Host ""
                Write-Host "[OK] Extraction successful!" -ForegroundColor Green
            } else {
                Write-Host ""
                Write-Host "[ERROR] Extraction failed! Exit code: $LASTEXITCODE" -ForegroundColor Red
                Pause
                Exit 1
            }
        } catch {
            Write-Host "[ERROR] tar command not available!" -ForegroundColor Red
            Write-Host "Error: $_" -ForegroundColor Red
            Write-Host ""
            Write-Host "Your Windows version may not support tar command." -ForegroundColor Yellow
            Write-Host "Please upgrade to Windows 10 1803+ or choose another method." -ForegroundColor Yellow
            Pause
            Exit 1
        }
    }
    
    "3" {
        # Use 7-Zip
        Write-Host ""
        Write-Host "[Method 3] Using 7-Zip to extract..." -ForegroundColor Cyan
        Write-Host ""
        
        # Find 7-Zip installation path
        $7zPaths = @(
            "C:\Program Files\7-Zip\7z.exe",
            "C:\Program Files (x86)\7-Zip\7z.exe"
        )
        
        $7zExe = $null
        foreach ($Path in $7zPaths) {
            if (Test-Path $Path) {
                $7zExe = $Path
                break
            }
        }
        
        if ($null -eq $7zExe) {
            Write-Host "[ERROR] 7-Zip not found!" -ForegroundColor Red
            Write-Host ""
            Write-Host "Please install 7-Zip: https://www.7-zip.org/" -ForegroundColor Yellow
            Write-Host "Or choose another extraction method." -ForegroundColor Yellow
            Pause
            Exit 1
        }
        
        Write-Host "[OK] Found 7-Zip: $7zExe" -ForegroundColor Green
        Write-Host "Extracting (this may take a few minutes)..." -ForegroundColor Yellow
        Write-Host ""
        
        # Extract using 7-Zip
        & $7zExe x $TarFile -so | & $7zExe x -si -ttar
        
        if ($LASTEXITCODE -eq 0) {
            # Rename directory
            if (Test-Path ".\fs-mol") {
                Rename-Item -Path ".\fs-mol" -NewName "fs-mol-dataset"
            }
            Write-Host ""
            Write-Host "[OK] Extraction successful!" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "[ERROR] Extraction failed! Exit code: $LASTEXITCODE" -ForegroundColor Red
            Pause
            Exit 1
        }
    }
    
    default {
        Write-Host ""
        Write-Host "[ERROR] Invalid selection!" -ForegroundColor Red
        Pause
        Exit 1
    }
}

# Verify extraction results
Write-Host ""
Write-Host "======================================"
Write-Host " Verifying Extraction Results"
Write-Host "======================================"
Write-Host ""

if (-Not (Test-Path $OutputDir)) {
    Write-Host "[ERROR] Output directory not found!" -ForegroundColor Red
    Pause
    Exit 1
}

# Check required subdirectories
$RequiredDirs = @("train", "valid", "test")
$AllExist = $true

foreach ($Dir in $RequiredDirs) {
    $DirPath = Join-Path $OutputDir $Dir
    if (Test-Path $DirPath) {
        $FileCount = (Get-ChildItem -Path $DirPath -Filter "*.jsonl.gz").Count
        Write-Host "[OK] $Dir/ directory exists ($FileCount files)" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] $Dir/ directory not found!" -ForegroundColor Red
        $AllExist = $false
    }
}

Write-Host ""
if ($AllExist) {
    Write-Host "======================================"
    Write-Host " Extraction Complete!"
    Write-Host "======================================"
    Write-Host ""
    Write-Host "Dataset successfully extracted to: $OutputDir" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Activate conda environment: conda activate adkf-ift-fsmol"
    Write-Host "2. Test loading dataset: python -c `"from fs_mol.data import FSMolDataset; FSMolDataset.from_directory('./fs-mol-dataset')`""
    Write-Host ""
} else {
    Write-Host "[WARNING] Extraction may be incomplete, please check!" -ForegroundColor Yellow
}

Pause
