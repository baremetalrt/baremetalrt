; BareMetalRT Installer — Inno Setup 6.1+
; Builds a modern, branded Windows installer.
;
; Build: "C:\Users\brian\AppData\Local\Programs\Inno Setup 6\ISCC.exe" baremetalrt.iss
; Or:    build-installer.ps1

#define MyAppName      "BareMetalRT"
#define MyAppVersion   "0.6.5-beta"
#define MyAppPublisher "Bare Metal AI, Inc."
#define MyAppURL       "https://baremetalrt.ai"
#define MyAppExeName   "baremetalrt.exe"

[Setup]
AppId={{7A2D4F8E-3B1C-4E5A-9D6F-2C8A1B0E4D3F}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
CloseApplications=force
RestartApplications=no
LicenseFile=assets\license.rtf
OutputDir=..\dist
OutputBaseFilename=BareMetalRT-{#MyAppVersion}-Setup
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes

; --- Modern layout (default size like Git/Notepad++) ---
WizardStyle=modern
WizardImageFile=assets\wizard-image.bmp
WizardSmallImageFile=assets\wizard-small.bmp

; --- Require 64-bit Windows ---
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0

; --- Privileges ---
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog

; --- Suppress per-user area warning (startup entry is intentional) ---
UsedUserAreasWarning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"
Name: "startupentry"; Description: "Start BareMetalRT when Windows starts"; GroupDescription: "Startup:"

[Files]
Source: "..\dist\baremetalrt.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\VERSION"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\daemon\build_engine.py"; DestDir: "{app}\daemon"; Flags: ignoreversion
Source: "..\daemon\model_registry.py"; DestDir: "{app}\daemon"; Flags: ignoreversion
Source: "..\daemon\convert_tp.py"; DestDir: "{app}\daemon"; Flags: ignoreversion
; Custom TRT-LLM Windows port — Python source only (libs/ DLLs are in runtime)
Source: "..\engine\tensorrt-llm\tensorrt_llm\*.py"; DestDir: "{app}\engine\tensorrt-llm\tensorrt_llm"; Flags: ignoreversion recursesubdirs
Source: "..\engine\tensorrt-llm\tensorrt_llm\commands\*"; DestDir: "{app}\engine\tensorrt-llm\tensorrt_llm\commands"; Flags: ignoreversion recursesubdirs; Excludes: "__pycache__,*.pyc"
Source: "..\engine\tensorrt-llm\tensorrt_llm\models\*"; DestDir: "{app}\engine\tensorrt-llm\tensorrt_llm\models"; Flags: ignoreversion recursesubdirs; Excludes: "__pycache__,*.pyc"
Source: "..\engine\tensorrt-llm\tensorrt_llm\plugin\*"; DestDir: "{app}\engine\tensorrt-llm\tensorrt_llm\plugin"; Flags: ignoreversion recursesubdirs; Excludes: "__pycache__,*.pyc"
Source: "..\engine\tensorrt-llm\tensorrt_llm\layers\*"; DestDir: "{app}\engine\tensorrt-llm\tensorrt_llm\layers"; Flags: ignoreversion recursesubdirs; Excludes: "__pycache__,*.pyc"
Source: "..\engine\tensorrt-llm\tensorrt_llm\quantization\*"; DestDir: "{app}\engine\tensorrt-llm\tensorrt_llm\quantization"; Flags: ignoreversion recursesubdirs; Excludes: "__pycache__,*.pyc"
Source: "..\engine\tensorrt-llm\tensorrt_llm\runtime\*"; DestDir: "{app}\engine\tensorrt-llm\tensorrt_llm\runtime"; Flags: ignoreversion recursesubdirs; Excludes: "__pycache__,*.pyc"
Source: "..\engine\tensorrt-llm\tensorrt_llm\builder.py"; DestDir: "{app}\engine\tensorrt-llm\tensorrt_llm"; Flags: ignoreversion
Source: "..\engine\tensorrt-llm\tensorrt_llm\functional.py"; DestDir: "{app}\engine\tensorrt-llm\tensorrt_llm"; Flags: ignoreversion

Source: "triton_kernels__init__.py"; DestDir: "{app}\engine\tensorrt-llm\triton_kernels"; DestName: "__init__.py"; Flags: ignoreversion
; Runtime DLLs — installed to %APPDATA%\BareMetalRT\runtime (user-writable)
Source: "..\runtime\bmrt_plugins_dll.dll"; DestDir: "{userappdata}\BareMetalRT\runtime"; Flags: ignoreversion
Source: "..\runtime\nvinfer_plugin_tensorrt_llm.dll"; DestDir: "{userappdata}\BareMetalRT\runtime"; Flags: ignoreversion
Source: "..\runtime\tensorrt_llm.dll"; DestDir: "{userappdata}\BareMetalRT\runtime"; Flags: ignoreversion

[Icons]
Name: "{group}\BareMetalRT"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\icon.ico"; Comment: "BareMetalRT GPU Inference Daemon"
Name: "{group}\Uninstall BareMetalRT"; Filename: "{uninstallexe}"
Name: "{autodesktop}\BareMetalRT"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon
Name: "{userstartup}\BareMetalRT"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--no-browser"; Tasks: startupentry

[Dirs]
Name: "{app}\engine_cache"; Permissions: users-modify
Name: "{app}\models"; Permissions: users-modify

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch BareMetalRT"; Flags: nowait postinstall skipifsilent

; Firewall cleanup handled in [Code] CurUninstallStepChanged to avoid visible windows

[Registry]
Root: HKLM; Subkey: "Software\BareMetalRT"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKLM; Subkey: "Software\BareMetalRT"; ValueType: string; ValueName: "Version"; ValueData: "{#MyAppVersion}"; Flags: uninsdeletekey

[Code]
// ─────────────────────────────────────────────────────
// Custom "NVIDIA Prerequisites" wizard page
// ─────────────────────────────────────────────────────

var
  PrereqPage: TWizardPage;

function DetectCUDA: Boolean;
var
  Versions: array of string;
  I: Integer;
  Path: String;
begin
  Result := False;
  SetArrayLength(Versions, 5);
  Versions[0] := '12.8'; Versions[1] := '12.6'; Versions[2] := '12.4';
  Versions[3] := '12.9'; Versions[4] := '12.5';
  for I := 0 to GetArrayLength(Versions) - 1 do
  begin
    Path := ExpandConstant('{pf}\NVIDIA GPU Computing Toolkit\CUDA\v' + Versions[I] + '\bin\nvcc.exe');
    if FileExists(Path) then
    begin
      Result := True;
      Exit;
    end;
  end;
end;

function DetectTensorRT: Boolean;
var
  Versions: array of string;
  I: Integer;
  Path: String;
begin
  Result := False;
  SetArrayLength(Versions, 4);
  Versions[0] := '10.15.1.29'; Versions[1] := '10.16.0.72';
  Versions[2] := '10.15.1'; Versions[3] := '10.15.0';
  for I := 0 to GetArrayLength(Versions) - 1 do
  begin
    Path := 'C:\TensorRT\TensorRT-' + Versions[I] + '\bin\nvinfer_10.dll';
    if FileExists(Path) then
    begin
      Result := True;
      Exit;
    end;
  end;
end;

procedure CudaLinkClick(Sender: TObject);
var
  ErrorCode: Integer;
begin
  ShellExec('open', 'https://developer.nvidia.com/cuda-downloads', '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode);
end;

procedure TrtLinkClick(Sender: TObject);
var
  ErrorCode: Integer;
begin
  ShellExec('open', 'https://developer.nvidia.com/tensorrt/download', '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode);
end;

var
  CudaStatusLabel: TLabel;
  TrtStatusLabel: TLabel;

procedure CreatePrereqPage;
var
  W, Y: Integer;
  Lbl: TLabel;
  Link: TNewStaticText;
  Divider: TBevel;
begin
  PrereqPage := CreateCustomPage(wpLicense,
    'NVIDIA SDK Prerequisites',
    'BareMetalRT requires the following NVIDIA SDKs for GPU inference.');

  W := PrereqPage.SurfaceWidth;
  Y := 0;

  // ── CUDA section ──────────────────────────────────
  Lbl := TLabel.Create(PrereqPage);
  Lbl.Parent := PrereqPage.Surface;
  Lbl.Caption := 'CUDA Toolkit 12.4+';
  Lbl.Font.Name := 'Segoe UI Semibold';
  Lbl.Font.Size := 11;
  Lbl.Font.Style := [fsBold];
  Lbl.Left := 0; Lbl.Top := Y; Lbl.Width := W;
  Y := Y + 26;

  CudaStatusLabel := TLabel.Create(PrereqPage);
  CudaStatusLabel.Parent := PrereqPage.Surface;
  CudaStatusLabel.Font.Name := 'Segoe UI';
  CudaStatusLabel.Font.Size := 9;
  CudaStatusLabel.Left := 0; CudaStatusLabel.Top := Y; CudaStatusLabel.Width := W;
  Y := Y + 20;

  Lbl := TLabel.Create(PrereqPage);
  Lbl.Parent := PrereqPage.Surface;
  Lbl.Caption := 'Install to the default path (Program Files\NVIDIA GPU Computing Toolkit).';
  Lbl.Font.Name := 'Segoe UI';
  Lbl.Font.Size := 9;
  Lbl.Font.Color := clGray;
  Lbl.Left := 0; Lbl.Top := Y; Lbl.Width := W;
  Y := Y + 20;

  Link := TNewStaticText.Create(PrereqPage);
  Link.Parent := PrereqPage.Surface;
  Link.Caption := 'Download CUDA Toolkit';
  Link.Font.Name := 'Segoe UI';
  Link.Font.Size := 9;
  Link.Font.Color := clBlue;
  Link.Font.Style := [fsUnderline];
  Link.Cursor := crHand;
  Link.OnClick := @CudaLinkClick;
  Link.Left := 0; Link.Top := Y; Link.Width := W;
  Y := Y + 28;

  // ── Divider ───────────────────────────────────────
  Divider := TBevel.Create(PrereqPage);
  Divider.Parent := PrereqPage.Surface;
  Divider.Shape := bsTopLine;
  Divider.Left := 0; Divider.Top := Y; Divider.Width := W; Divider.Height := 2;
  Y := Y + 12;

  // ── TensorRT section ──────────────────────────────
  Lbl := TLabel.Create(PrereqPage);
  Lbl.Parent := PrereqPage.Surface;
  Lbl.Caption := 'TensorRT 10.15+';
  Lbl.Font.Name := 'Segoe UI Semibold';
  Lbl.Font.Size := 11;
  Lbl.Font.Style := [fsBold];
  Lbl.Left := 0; Lbl.Top := Y; Lbl.Width := W;
  Y := Y + 26;

  TrtStatusLabel := TLabel.Create(PrereqPage);
  TrtStatusLabel.Parent := PrereqPage.Surface;
  TrtStatusLabel.Font.Name := 'Segoe UI';
  TrtStatusLabel.Font.Size := 9;
  TrtStatusLabel.Left := 0; TrtStatusLabel.Top := Y; TrtStatusLabel.Width := W;
  Y := Y + 20;

  Lbl := TLabel.Create(PrereqPage);
  Lbl.Parent := PrereqPage.Surface;
  Lbl.Caption := 'Extract the zip to C:\TensorRT\ (keep the versioned folder inside).';
  Lbl.Font.Name := 'Segoe UI';
  Lbl.Font.Size := 9;
  Lbl.Font.Color := clGray;
  Lbl.Left := 0; Lbl.Top := Y; Lbl.Width := W;
  Y := Y + 20;

  Link := TNewStaticText.Create(PrereqPage);
  Link.Parent := PrereqPage.Surface;
  Link.Caption := 'Download TensorRT';
  Link.Font.Name := 'Segoe UI';
  Link.Font.Size := 9;
  Link.Font.Color := clBlue;
  Link.Font.Style := [fsUnderline];
  Link.Cursor := crHand;
  Link.OnClick := @TrtLinkClick;
  Link.Left := 0; Link.Top := Y; Link.Width := W;
  Y := Y + 28;

  // ── Divider ───────────────────────────────────────
  Divider := TBevel.Create(PrereqPage);
  Divider.Parent := PrereqPage.Surface;
  Divider.Shape := bsTopLine;
  Divider.Left := 0; Divider.Top := Y; Divider.Width := W; Divider.Height := 2;
  Y := Y + 16;

  // ── Info note ─────────────────────────────────────
  Lbl := TLabel.Create(PrereqPage);
  Lbl.Parent := PrereqPage.Surface;
  Lbl.Caption :=
    'You can install BareMetalRT now and add the NVIDIA SDKs later.'
    + #13#10 + 'The daemon will detect them automatically on startup.'
    + #13#10#13#10 + 'NVIDIA downloads require a free Developer account.';
  Lbl.Font.Name := 'Segoe UI';
  Lbl.Font.Size := 9;
  Lbl.Font.Color := clGray;
  Lbl.Font.Style := [fsItalic];
  Lbl.WordWrap := True;
  Lbl.Left := 0; Lbl.Top := Y; Lbl.Width := W; Lbl.Height := 60;
end;

procedure UpdatePrereqStatus;
begin
  if DetectCUDA then
  begin
    CudaStatusLabel.Caption := 'Status:  Detected';
    CudaStatusLabel.Font.Color := clGreen;
  end else
  begin
    CudaStatusLabel.Caption := 'Status:  Not detected';
    CudaStatusLabel.Font.Color := clRed;
  end;

  if DetectTensorRT then
  begin
    TrtStatusLabel.Caption := 'Status:  Detected';
    TrtStatusLabel.Font.Color := clGreen;
  end else
  begin
    TrtStatusLabel.Caption := 'Status:  Not detected';
    TrtStatusLabel.Font.Color := clRed;
  end;
end;

// ─────────────────────────────────────────────────────
// Firewall rules — added after install completes
// ─────────────────────────────────────────────────────
procedure AddFirewallRules;
var
  ResultCode: Integer;
begin
  Exec('netsh', 'advfirewall firewall add rule name="BareMetalRT" dir=in action=allow protocol=TCP localport=8080',
       '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Exec('netsh', 'advfirewall firewall add rule name="BareMetalRT-TCP-Transport" dir=in action=allow protocol=TCP localport=29500-29610',
       '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

// ─────────────────────────────────────────────────────
// Default config file
// ─────────────────────────────────────────────────────
procedure CreateDefaultConfig;
var
  ConfigDir, ConfigPath: String;
begin
  ConfigDir := ExpandConstant('{userappdata}\BareMetalRT');
  ConfigPath := ConfigDir + '\config.json';
  if not FileExists(ConfigPath) then
  begin
    ForceDirectories(ConfigDir);
    SaveStringToFile(ConfigPath,
      '{' + #13#10 +
      '    "api_key": "",' + #13#10 +
      '    "orchestrator": "https://baremetalrt.ai"' + #13#10 +
      '}' + #13#10, False);
  end;
end;

// ─────────────────────────────────────────────────────
// Event hooks
// ─────────────────────────────────────────────────────
procedure InitializeWizard;
begin
  CreatePrereqPage;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = PrereqPage.ID then
    UpdatePrereqStatus;
end;

procedure RemoveFirewallRules;
var
  ResultCode: Integer;
begin
  Exec('netsh', 'advfirewall firewall delete rule name="BareMetalRT"',
       '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Exec('netsh', 'advfirewall firewall delete rule name="BareMetalRT-TCP-Transport"',
       '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
  FindRec: TFindRec;
  TempDir, Path: String;
begin
  Result := '';
  NeedsRestart := False;

  // Kill running BareMetalRT and any spawned processes
  Exec('taskkill.exe', '/F /IM baremetalrt.exe /T', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Sleep(500);
  // Kill again in case tray respawned
  Exec('taskkill.exe', '/F /IM baremetalrt.exe /T', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Sleep(2000);

  // Clean up stale _MEI* dirs from PyInstaller
  TempDir := ExpandConstant('{localappdata}\Temp\');
  if FindFirst(TempDir + '_MEI*', FindRec) then
  begin
    try
      repeat
        Path := TempDir + FindRec.Name;
        if FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0 then
          DelTree(Path, True, True, True);
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    AddFirewallRules;
    CreateDefaultConfig;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFirewallRules;
end;
