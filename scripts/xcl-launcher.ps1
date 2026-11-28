[CmdletBinding()]
param(
    [switch]$Launch,
    [string]$Game,
    [string]$Xcl,
    [string]$Preset,
    [hashtable]$Options = @{}
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\xcl-launcher-options.ps1"
$catalog = @(Get-XclOptions -SourceRoot (Split-Path -Parent $PSScriptRoot) |
    Where-Object Area -ne 'Shader CLI')
$byName = @{}
foreach ($option in $catalog) { $byName[$option.Name] = $option }
$builtIn = @{
    Default = @{}
    Debugging = @{ XCL_ACTIVITY = '1'; XCL_TRACE_FAULTS = '1'; XCL_THREAD_DUMP_AT = '30,60,120' }
    '60-second smoke' = @{ XCL_ACTIVITY = '1'; XCL_RUN_SECONDS = '60'; XCL_NETWORK = '0' }
}
$settingsFile = Join-Path $env:LOCALAPPDATA 'XCL\developer-launcher.json'

function Read-Options($Object) {
    $map = @{}
    foreach ($property in $Object.PSObject.Properties) {
        if ($property.Name -cmatch '^XCL_[A-Z0-9_]+$' -and $property.Name -ne 'XCL_HOST' -and
            $property.Value -is [string] -and $property.Value -ne '') {
            $map[$property.Name] = $property.Value
        }
    }
    $map
}

$state = @{ Game = ''; Xcl = ''; Preset = 'Default'; Options = @{}; Presets = @{} }
if ([IO.File]::Exists($settingsFile)) {
    $saved = [IO.File]::ReadAllText($settingsFile) | ConvertFrom-Json
    foreach ($key in 'Game','Xcl','Preset') {
        if ($saved.$key -is [string]) { $state[$key] = $saved.$key }
    }
    $state.Options = Read-Options $saved.Options
    foreach ($property in $saved.Presets.PSObject.Properties) {
        if (-not $builtIn.ContainsKey($property.Name)) {
            $state.Presets[$property.Name] = Read-Options $property.Value
        }
    }
}

function Save-Settings {
    $dir = [IO.Path]::GetDirectoryName($settingsFile)
    $null = [IO.Directory]::CreateDirectory($dir)
    $temp = Join-Path $dir ([IO.Path]::GetRandomFileName())
    try {
        [IO.File]::WriteAllText($temp, ($state | ConvertTo-Json -Depth 6))
        if ([IO.File]::Exists($settingsFile)) {
            [IO.File]::Replace($temp, $settingsFile, ($settingsFile + '.bak'))
        } else { [IO.File]::Move($temp, $settingsFile) }
    } finally { [IO.File]::Delete($temp) }
}

function Use-Preset([string]$Name) {
    $values = if ($builtIn.ContainsKey($Name)) { $builtIn[$Name] } else { $state.Presets[$Name] }
    if ($null -eq $values) { throw "Unknown preset: $Name" }
    $state.Options = $values.Clone()
    $state.Preset = $Name
}

function Start-Title {
    if (-not $state.Game -or -not $state.Xcl) { throw 'Select an XCL executable and a game location first.' }
    $exe = Get-Item -LiteralPath $state.Xcl
    $target = Get-Item -LiteralPath $state.Game
    if ($exe -isnot [IO.FileInfo] -or $exe.Extension -ne '.exe') { throw 'Select an XCL executable.' }
    if ($target -is [IO.FileInfo] -and $target.Name -eq 'MicrosoftGame.config') { $target = $target.Directory }
    if ($target -isnot [IO.DirectoryInfo] -and $target.Extension -ne '.exe') {
        throw 'Select a game folder, executable, or MicrosoftGame.config.'
    }
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $exe.FullName
    $info.WorkingDirectory = $exe.DirectoryName
    # Windows paths cannot contain quotes. Double trailing backslashes inside a quoted argument.
    $info.Arguments = 'run "' + ($target.FullName -replace '(\\+)$', '$1$1') + '"'
    $info.UseShellExecute = $false
    foreach ($name in $byName.Keys) { $info.EnvironmentVariables.Remove($name) }
    $info.EnvironmentVariables.Remove('XCL_HOST')
    foreach ($name in $state.Options.Keys) {
        if (-not $byName.ContainsKey($name)) { continue }
        $value = [string]$state.Options[$name]
        if ($value -eq '') { continue }
        $kind = $byName[$name].Kind
        if (($kind -eq 'Flag' -and $value -ne '1') -or
            ($kind -eq 'Boolean' -and $value -notin '0','1') -or $value.Contains([char]0)) {
            throw "Invalid value for $name ($kind). Use 1 for a flag or 0/1 for a Boolean."
        }
        $info.EnvironmentVariables[$name] = $value
    }
    $state.Xcl = $exe.FullName
    $state.Game = $target.FullName
    Save-Settings
    [Diagnostics.Process]::Start($info)
}

if ($Game) { $state.Game = $Game }
if ($Xcl) { $state.Xcl = $Xcl }
if ($Preset) { Use-Preset $Preset }
foreach ($name in $Options.Keys) {
    if (-not $byName.ContainsKey($name)) { throw "Unknown launch option: $name" }
    if ([string]$Options[$name] -eq '') { $state.Options.Remove($name) }
    else { $state.Options[$name] = [string]$Options[$name] }
}
if ($Launch) {
    $process = Start-Title
    $process.WaitForExit()
    exit $process.ExitCode
}

Add-Type -AssemblyName System.Windows.Forms,System.Drawing,Microsoft.VisualBasic
[Windows.Forms.Application]::EnableVisualStyles()
$form = New-Object Windows.Forms.Form -Property @{
    Text = 'XCL Developer Launcher'; StartPosition = 'CenterScreen'
    Size = [Drawing.Size]::new(1060,690); MinimumSize = [Drawing.Size]::new(860,480)
}
$layout = New-Object Windows.Forms.TableLayoutPanel -Property @{ Dock = 'Fill'; ColumnCount = 1; RowCount = 6 }
foreach ($height in 39,39,43,42) { $null = $layout.RowStyles.Add([Windows.Forms.RowStyle]::new('Absolute',$height)) }
$null = $layout.RowStyles.Add([Windows.Forms.RowStyle]::new('Percent',100))
$null = $layout.RowStyles.Add([Windows.Forms.RowStyle]::new('Absolute',42))
$form.Controls.Add($layout)

function New-Bar([int]$Row) {
    $bar = New-Object Windows.Forms.FlowLayoutPanel -Property @{ Dock = 'Fill'; WrapContents = $false }
    $layout.Controls.Add($bar,0,$Row)
    $bar
}
function Add-Button($Bar, [string]$Text, [scriptblock]$Action) {
    $button = New-Object Windows.Forms.Button -Property @{ Text = $Text; AutoSize = $true }
    $button.Tag = $Action
    $button.Add_Click({ param($sender) try { & $sender.Tag } catch { Show-Error $_ } })
    $Bar.Controls.Add($button)
}
function Show-Error($Failure) {
    $null = [Windows.Forms.MessageBox]::Show($form,$Failure.Exception.Message,'XCL Developer Launcher')
}
function Pick-File($Box, [string]$Filter) {
    $dialog = New-Object Windows.Forms.OpenFileDialog -Property @{ Filter = $Filter; CheckFileExists = $true }
    try {
        if ($Box.Text.Trim() -and (Test-Path -LiteralPath $Box.Text -PathType Leaf)) {
            $dialog.FileName = $Box.Text
        }
        if ($dialog.ShowDialog($form) -eq 'OK') { $Box.Text = $dialog.FileName }
    } finally { $dialog.Dispose() }
}
function Add-PathRow([int]$Row, [string]$Label, [string]$Value) {
    $bar = New-Bar $Row
    $bar.Controls.Add((New-Object Windows.Forms.Label -Property @{ Text = $Label; Width = 85 }))
    $box = New-Object Windows.Forms.TextBox -Property @{ Text = $Value; Width = 475 }
    $bar.Controls.Add($box)
    @($bar,$box)
}

$bar,$exeBox = Add-PathRow 0 'XCL executable' $state.Xcl
Add-Button $bar 'Browse...' { Pick-File $exeBox 'Executables (*.exe)|*.exe' }
$bar,$gameBox = Add-PathRow 1 'Game location' $state.Game
Add-Button $bar 'Game file...' { Pick-File $gameBox 'Game executable or manifest|*.exe;MicrosoftGame.config' }
Add-Button $bar 'Game folder...' {
    $dialog = New-Object Windows.Forms.FolderBrowserDialog
    try {
        if ($gameBox.Text.Trim() -and (Test-Path -LiteralPath $gameBox.Text -PathType Container)) {
            $dialog.SelectedPath = $gameBox.Text
        }
        if ($dialog.ShowDialog($form) -eq 'OK') { $gameBox.Text = $dialog.SelectedPath }
    } finally { $dialog.Dispose() }
}
$bar = New-Bar 2
$picker = New-Object Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 170; AccessibleName = 'Preset' }
$bar.Controls.Add($picker)
function Refresh-Presets {
    $picker.Items.Clear()
    foreach ($name in @($builtIn.Keys) + @($state.Presets.Keys) | Sort-Object) { $null = $picker.Items.Add($name) }
    if ($picker.Items.Contains($state.Preset)) { $picker.SelectedItem = $state.Preset } else { $picker.SelectedItem = 'Default' }
}
Add-Button $bar 'Apply preset' { Sync-Location; Use-Preset $picker.Text; Refresh-Grid; Save-Settings }
Add-Button $bar 'Save preset as...' {
    Sync-Location
    $name = [Microsoft.VisualBasic.Interaction]::InputBox('Preset name','Save preset',$state.Preset).Trim()
    if (-not $name) { return }
    if ($builtIn.ContainsKey($name)) { throw 'Choose a new name. Built-in presets are read-only.' }
    if ($state.Presets.ContainsKey($name) -and
        [Windows.Forms.MessageBox]::Show($form,"Replace preset '$name'?",'Save preset','YesNo') -ne 'Yes') { return }
    $state.Presets[$name] = $state.Options.Clone()
    $state.Preset = $name
    Save-Settings
    Refresh-Presets
}
Add-Button $bar 'Delete preset' {
    $name = $picker.Text
    if (-not $state.Presets.ContainsKey($name)) { throw 'Built-in presets cannot be deleted.' }
    if ([Windows.Forms.MessageBox]::Show($form,"Delete preset '$name'?",'Delete preset','YesNo') -ne 'Yes') { return }
    $state.Presets.Remove($name)
    if ($state.Preset -eq $name) { $state.Preset = 'Default' }
    Sync-Location
    Save-Settings
    Refresh-Presets
}
$bar = New-Bar 3
$bar.Controls.Add((New-Object Windows.Forms.Label -Property @{ Text = 'Find option'; AutoSize = $true }))
$search = New-Object Windows.Forms.TextBox -Property @{ Width = 245 }
$bar.Controls.Add($search)
$bar.Controls.Add((New-Object Windows.Forms.Label -Property @{ Text = 'Blank uses XCL defaults. Flags accept 1. Boolean values accept 0 or 1.'; AutoSize = $true }))
$grid = New-Object Windows.Forms.DataGridView -Property @{
    Dock = 'Fill'; AutoSizeColumnsMode = 'Fill'; AllowUserToAddRows = $false
    AllowUserToDeleteRows = $false; RowHeadersVisible = $false
}
foreach ($entry in @(@('Option',26),@('Value',14),@('Area',12),@('Description',48))) {
    $column = New-Object Windows.Forms.DataGridViewTextBoxColumn -Property @{
        Name = $entry[0]; HeaderText = $entry[0]; FillWeight = $entry[1]; ReadOnly = ($entry[0] -ne 'Value')
    }
    $null = $grid.Columns.Add($column)
}
$layout.Controls.Add($grid,0,4)
function Refresh-Grid {
    $null = $grid.EndEdit()
    $grid.Rows.Clear()
    foreach ($opt in $catalog) {
        if ($search.Text -and "$($opt.Name) $($opt.Area) $($opt.Description)" -notmatch [regex]::Escape($search.Text)) { continue }
        $null = $grid.Rows.Add($opt.Name, [string]$state.Options[$opt.Name], $opt.Area, $opt.Description)
    }
}
$grid.Add_CellEndEdit({
    param($sender,$e)
    if ($e.RowIndex -lt 0 -or $e.ColumnIndex -ne 1) { return }
    $cells = $grid.Rows[$e.RowIndex].Cells
    $name = [string]$cells['Option'].Value
    $value = [string]$cells['Value'].Value
    if ($value -ne '') { $state.Options[$name] = $value } else { $state.Options.Remove($name) }
})
$search.Add_TextChanged({ Refresh-Grid })

function Sync-Location {
    $null = $grid.EndEdit()
    $state.Xcl = $exeBox.Text.Trim()
    $state.Game = $gameBox.Text.Trim()
}
$bar = New-Bar 5
Add-Button $bar 'Launch game' { Sync-Location; $null = Start-Title }
Add-Button $bar 'Open XCL data...' {
    Sync-Location
    $dir = $state.Options['XCL_DATA_ROOT']
    if (-not $dir) { $dir = Join-Path $env:LOCALAPPDATA 'XCL' }
    if (-not (Test-Path -LiteralPath $dir -PathType Container)) { throw "XCL creates this folder on launch: $dir" }
    Invoke-Item -LiteralPath $dir
}
$form.Add_FormClosing({
    param($sender,$e)
    try { Sync-Location; Save-Settings } catch { $e.Cancel = $true; Show-Error $_ }
})
Refresh-Presets
Refresh-Grid
try { $null = $form.ShowDialog() } finally { $form.Dispose() }
