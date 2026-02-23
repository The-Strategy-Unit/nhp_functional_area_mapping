$ErrorActionPreference = "Stop"

$EnvFile = ".env"

if (-not (Test-Path $EnvFile)) {
    throw ".env file not found at path: $EnvFile"
}

Get-Content $EnvFile | ForEach-Object {
    if ($_ -match '^\s*#' -or [string]::IsNullOrWhiteSpace($_)) {
        return
    }

    $name, $value = $_ -split '=', 2

    $name  = $name.Trim()
    $value = $value.Trim()

    Set-Item -Path "Env:$name" -Value $value
}

# --- Read required variables ---
$StorageAccountName = $env:AZURE_STORAGE_ACCOUNT_NAME
$TableName          = $env:TABLE_NAME
$ContainerName      = $env:STORAGE_CONTAINER_NAME
$SecretScope        = $env:DATABRICKS_SECRET_SCOPE
$SecretTableKey     = $env:DATABRICKS_SECRET_TABLE_KEY
$SecretStorageKey   = $env:DATABRICKS_SECRET_STORAGE_KEY

foreach ($var in @(
    "STORAGE_ACCOUNT_NAME",
    "TABLE_NAME",
    "DATABRICKS_SECRET_SCOPE",
    "DATABRICKS_SECRET_TABLE_KEY",
    "DATABRICKS_SECRET_STORAGE_KEY"
)) {
    if (-not $var) {
        throw "Missing required environment variable: $var"
    }
}

$ExpiryDate = (Get-Date).ToUniversalTime().AddDays(1).ToString("yyyy-MM-ddTHH:mmZ")

Write-Host "Generating Table SAS token (expires $ExpiryDate)..."

$TableToken = az storage table generate-sas `
    --account-name $StorageAccountName `
    --name $TableName `
    --permissions aur `
    --expiry $ExpiryDate `
    --output tsv

if (-not $TableToken) {
    throw "Failed to generate Table SAS token"
}

Write-Host "Table SAS token generated successfully."

Write-Host "Uploading Table SAS token to Databricks Secrets..."

$TableToken | databricks secrets put-secret `
    $SecretScope `
    $SecretTableKey

Write-Host "✅ Table SAS stored in Databricks secret scope '$SecretScope' as '$SecretTableKey'"

Write-Host "Generating Storage SAS token (expires $ExpiryDate)..."

$StorageToken = az storage container generate-sas `
    --account-name $StorageAccountName `
    --name $ContainerName `
    --permissions aclrw `
    --expiry $ExpiryDate `
    --auth-mode login `
    --as-user `
    --output tsv

if (-not $StorageToken) {
    throw "Failed to generate Storage SAS token"
}

Write-Host "Storage SAS token generated successfully."

Write-Host "Uploading Storage SAS token to Databricks Secrets..."

$StorageToken | databricks secrets put-secret `
    $SecretScope `
    $SecretStorageKey

Write-Host "✅ Storage SAS stored in Databricks secret scope '$SecretScope' as '$SecretStorageKey'"