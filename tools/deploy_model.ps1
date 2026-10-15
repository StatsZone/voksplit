# tools/deploy_model.ps1 - Automatisation du déploiement de VoksplitNet FP16
param (
    [string]$SourceFile = "$env:USERPROFILE\Downloads\voksplit-net-fp16.onnx"
)

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  DEPLOIEMENT DU MODELE VOKSPLITNET FP16 IA LOCALE   " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

if (-not (Test-Path $SourceFile)) {
    Write-Host "[ERREUR] Le fichier $SourceFile est introuvable." -ForegroundColor Red
    Write-Host "Veuillez télécharger voksplit-net-fp16.onnx depuis Google Colab ou spécifier le chemin." -ForegroundColor Yellow
    exit 1
}

$fileSize = (Get-Item $SourceFile).Length / 1MB
Write-Host "[1/5] Fichier source trouvé : $SourceFile ($([Math]::Round($fileSize, 2)) Mo)" -ForegroundColor Green

# Calcul du Hash SHA-256
$hash = (Get-FileHash -Path $SourceFile -Algorithm SHA256).Hash.ToLower()
Write-Host "[2/5] Empreinte SHA-256 calculée : $hash" -ForegroundColor Green

# Chemins de destination
$destVoksplit = "C:\Users\Flo\Desktop\Voksplit\public\models\voksplit-net-fp16.onnx"
$destSoundCapsule = "C:\Users\Flo\Desktop\Sound Capsule\models\voksplit-net-fp16.onnx"

# Copie vers Voksplit
New-Item -ItemType Directory -Force -Path "C:\Users\Flo\Desktop\Voksplit\public\models" | Out-Null
Copy-Item -Path $SourceFile -Destination $destVoksplit -Force
Write-Host "[3/5] Modèle copié vers Voksplit : $destVoksplit" -ForegroundColor Green

# Copie vers Sound Capsule
New-Item -ItemType Directory -Force -Path "C:\Users\Flo\Desktop\Sound Capsule\models" | Out-Null
Copy-Item -Path $SourceFile -Destination $destSoundCapsule -Force
Write-Host "[3/5] Modèle copié vers Sound Capsule : $destSoundCapsule" -ForegroundColor Green

# Injection du SHA-256 dans les workers
$workerVoksplit = "C:\Users\Flo\Desktop\Voksplit\public\stem-worker.js"
$workerSC = "C:\Users\Flo\Desktop\Sound Capsule\stem-worker.js"

if (Test-Path $workerVoksplit) {
    (Get-Content $workerVoksplit) -replace "let VOKSPLIT_MODEL_SHA256 = '.*?';", "let VOKSPLIT_MODEL_SHA256 = '$hash';" | Set-Content $workerVoksplit
    Write-Host "[4/5] SHA-256 injecté dans Voksplit stem-worker.js" -ForegroundColor Green
}

if (Test-Path $workerSC) {
    (Get-Content $workerSC) -replace "let VOKSPLIT_MODEL_SHA256 = '.*?';", "let VOKSPLIT_MODEL_SHA256 = '$hash';" | Set-Content $workerSC
    Write-Host "[4/5] SHA-256 injecté dans Sound Capsule stem-worker.js" -ForegroundColor Green
}

Write-Host "[5/5] Modèle déployé avec succès sur les deux projets !" -ForegroundColor Cyan
Write-Host "Hash : $hash" -ForegroundColor Yellow
Write-Host "Prêt pour 'git add', 'npm run build' et 'git push'." -ForegroundColor Cyan
