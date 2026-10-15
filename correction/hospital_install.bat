@echo off
setlocal enabledelayedexpansion
title TCM ASR 一键式离线部署 — 医院终端版

:: ============================================================================
:: hospital_install.bat
:: ============================================================================
:: 功能：在完全断网的医院诊室终端上，一键完成以下操作：
::   1. 静默安装 Ollama（无弹窗、无需人工点击）
::   2. 离线导入 qwen2.5:1.5b 中医纠错模型
::   3. 配置 Ollama 开机自启并完全隐藏（对医生无感）
::   4. 验证模型可用性
::
:: 用法：将本脚本 + OllamaSetup.exe + ollama_model_offline\ 放入同一目录
::       双击 hospital_install.bat 即可（约需 2-3 分钟）
:: ============================================================================

set "SCRIPT_DIR=%~dp0"
set "SETUP=%SCRIPT_DIR%OllamaSetup.exe"
set "MODEL_SRC=%SCRIPT_DIR%ollama_model_offline"
set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
set "OLLAMA_APP_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
set "MODEL_DST=%USERPROFILE%\.ollama"
set "CONFIG_DST=%LOCALAPPDATA%\Ollama"
set "VBS_HIDDEN=%USERPROFILE%\ollama_hidden.vbs"

echo.
echo  ======================================================================
echo    TCM ASR 中医语音纠错模块 — 离线部署脚本
echo    医院终端专用版  v1.0
echo  ======================================================================
echo.
echo    目标模型 : qwen2.5:1.5b
echo    API 端口 : 11434
echo    部署模式 : 完全离线 / 开机自启 / 无感隐藏
echo.

:: =========================================================================
:: Step 0: 预检 — 确认部署物料齐全
:: =========================================================================
echo  [0/9] 检查部署物料...

if not exist "%SETUP%" (
    echo  [ERROR] 未找到 OllamaSetup.exe，请将其放入当前目录后重试。
    echo          当前目录: %SCRIPT_DIR%
    pause
    exit /b 1
)

if not exist "%MODEL_SRC%\models\blobs\" (
    echo  [ERROR] 未找到离线模型包，请先运行 prepare_offline_model.bat 生成。
    echo          需要目录: %MODEL_SRC%
    pause
    exit /b 1
)

echo         OllamaSetup.exe .......... [OK]
echo         ollama_model_offline\ .... [OK]

:: =========================================================================
:: Step 1: 关闭所有正在运行的 Ollama 进程
:: =========================================================================
echo  [1/9] 关闭现有 Ollama 进程...
taskkill /f /im "ollama app.exe"  >nul 2>&1
taskkill /f /im "ollama.exe"      >nul 2>&1
taskkill /f /im "OllamaSetup.exe" >nul 2>&1
timeout /t 2 /nobreak >nul
echo         Ollama 进程已清理。

:: =========================================================================
:: Step 2: 静默安装 Ollama
:: =========================================================================
echo  [2/9] 静默安装 Ollama（全程无弹窗，请耐心等待 30~60 秒）...
start /wait "" "%SETUP%" /silent
echo         Ollama 安装完成。

:: =========================================================================
:: Step 3: 等待安装写入磁盘并定位 ollama.exe
:: =========================================================================
echo  [3/9] 等待安装就绪...
set "OLLAMA_FOUND=0"
for /l %%i in (1,1,30) do (
    if exist "%OLLAMA_EXE%" (
        set "OLLAMA_FOUND=1"
        goto :ollama_ready
    )
    timeout /t 2 /nobreak >nul
    echo         等待中... (%%i/30)
)
:: 尝试系统级安装路径
if exist "C:\Program Files\Ollama\ollama.exe" (
    set "OLLAMA_EXE=C:\Program Files\Ollama\ollama.exe"
    set "OLLAMA_APP_EXE=C:\Program Files\Ollama\ollama app.exe"
    set "OLLAMA_FOUND=1"
)
:ollama_ready
if "!OLLAMA_FOUND!"=="0" (
    echo  [ERROR] 安装后未找到 ollama.exe，请检查是否被杀毒软件拦截。
    pause
    exit /b 1
)
echo         ollama.exe 已就绪: !OLLAMA_EXE!

:: =========================================================================
:: Step 4: 杀掉安装程序自动启动的 Ollama（准备导入模型）
:: =========================================================================
echo  [4/9] 暂停自动启动的 Ollama 服务...
taskkill /f /im "ollama app.exe" >nul 2>&1
taskkill /f /im "ollama.exe"     >nul 2>&1
timeout /t 3 /nobreak >nul
echo         Ollama 已暂停。

:: =========================================================================
:: Step 5: 离线导入模型文件
:: =========================================================================
echo  [5/9] 离线导入 qwen2.5:1.5b 模型文件...
echo         正在拷贝模型权重 (~941MB)，请稍候...

:: 确保目标目录存在
if not exist "%MODEL_DST%" mkdir "%MODEL_DST%"

:: 拷贝 blobs（模型权重）
if exist "%MODEL_SRC%\models\blobs\" (
    xcopy /e /i /q /y "%MODEL_SRC%\models\blobs\*" "%MODEL_DST%\models\blobs\" >nul
    if errorlevel 1 (
        echo  [ERROR] 拷贝模型权重失败！请检查磁盘空间（需至少 1GB 可用）。
        pause
        exit /b 1
    )
)

:: 拷贝 manifests（模型清单）
if exist "%MODEL_SRC%\models\manifests\" (
    xcopy /e /i /q /y "%MODEL_SRC%\models\manifests\*" "%MODEL_DST%\models\manifests\" >nul
)

:: 拷贝 Ollama 配置数据库（模型注册信息）
if exist "%MODEL_SRC%\config\" (
    for %%f in ("%MODEL_SRC%\config\*") do (
        copy /y "%%f" "%CONFIG_DST%\" >nul 2>&1
    )
    :: 同时拷贝身份密钥到 .ollama 目录
    if exist "%MODEL_SRC%\config\id_ed25519"     copy /y "%MODEL_SRC%\config\id_ed25519"     "%MODEL_DST%\" >nul 2>&1
    if exist "%MODEL_SRC%\config\id_ed25519.pub" copy /y "%MODEL_SRC%\config\id_ed25519.pub" "%MODEL_DST%\" >nul 2>&1
)

echo         模型文件导入完成。

:: =========================================================================
:: Step 6: 移除 Ollama 托盘图标程序的自启动（对医生屏蔽）
:: =========================================================================
echo  [6/9] 配置无感运行（屏蔽托盘图标）...

:: 方法1: 删除 / 禁用 Start Menu\Startup 中的快捷方式
if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Ollama.lnk" (
    del /f /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Ollama.lnk" >nul 2>&1
)
if exist "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Startup\Ollama.lnk" (
    del /f /q "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Startup\Ollama.lnk" >nul 2>&1
)

:: 方法2: 删除注册表中的 Run 键（Ollama 可能在 HKCU 或 HKLM 下注册）
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v "Ollama" /f >nul 2>&1
reg delete "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run" /v "Ollama" /f >nul 2>&1

:: 方法3: 禁用计划任务（如果存在）
schtasks /change /tn "Ollama" /disable >nul 2>&1

echo         托盘图标已屏蔽。

:: =========================================================================
:: Step 7: 创建后台隐藏自启服务
:: =========================================================================
echo  [7/9] 创建开机自启后台服务（完全隐藏窗口）...

:: 生成 VBS 脚本 —— 用 WScript.Shell.Run 的第二个参数 0 实现完全隐藏
(
echo ' Ollama Hidden Background Service
echo ' 由 hospital_install.bat 自动生成
echo On Error Resume Next
echo Dim WshShell
echo Set WshShell = CreateObject^("Wscript.Shell"^)
echo WshShell.Run """" ^& WshShell.ExpandEnvironmentStrings^("%%LOCALAPPDATA%%\Programs\Ollama\ollama.exe"^) ^& """ serve"", 0, False
echo Set WshShell = Nothing
) > "%VBS_HIDDEN%"

:: 移除旧的同名计划任务（如果存在）
schtasks /delete /tn "TCM-Ollama-Background" /f >nul 2>&1

:: 创建计划任务：用户登录时自动运行，窗口隐藏
schtasks /create ^
    /tn "TCM-Ollama-Background" ^
    /tr "wscript.exe \"%VBS_HIDDEN%\"" ^
    /sc onlogon ^
    /it ^
    /f ^
    /rl limited ^
    >nul 2>&1

if errorlevel 1 (
    echo  [WARN]  计划任务创建失败，改用注册表 Run 键方式...
    :: 降级方案：添加到注册表 Run
    reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" ^
        /v "TCM-Ollama-Background" ^
        /t REG_SZ ^
        /d "wscript.exe \"%VBS_HIDDEN%\"" ^
        /f >nul 2>&1
    echo         已通过注册表 Run 键配置自启。
) else (
    echo         计划任务已创建: TCM-Ollama-Background
)

:: =========================================================================
:: Step 8: 立即启动 Ollama 后台服务（无需重启）
:: =========================================================================
echo  [8/9] 启动 Ollama 后台服务...

:: 直接用 VBS 启动，窗口完全隐藏
wscript.exe "%VBS_HIDDEN%"

:: 等待服务就绪（最多等 30 秒）
echo         等待 Ollama 服务就绪...
set "SERVICE_READY=0"
for /l %%i in (1,1,30) do (
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if not errorlevel 1 (
        set "SERVICE_READY=1"
        goto :service_ready
    )
    timeout /t 2 /nobreak >nul
)
:service_ready
if "!SERVICE_READY!"=="0" (
    echo  [ERROR] Ollama 服务启动超时，请检查系统环境。
    pause
    exit /b 1
)
echo         Ollama 服务已就绪 (http://localhost:11434)。

:: =========================================================================
:: Step 9: 验证模型可用性
:: =========================================================================
echo  [9/9] 验证模型可用性...

:: 使用 REST API 检查模型列表
curl -s http://localhost:11434/api/tags 2>&1 | findstr /c:"qwen2.5:1.5b" >nul
if errorlevel 1 (
    echo  [WARN]  模型未在 API 中找到，尝试通过文件注册...
    :: 有时需要重启 Ollama 才能识别新模型
    taskkill /f /im "ollama.exe" >nul 2>&1
    timeout /t 3 /nobreak >nul
    wscript.exe "%VBS_HIDDEN%"
    timeout /t 10 /nobreak >nul

    curl -s http://localhost:11434/api/tags 2>&1 | findstr /c:"qwen2.5:1.5b" >nul
    if errorlevel 1 (
        echo  [ERROR] 模型验证失败。请尝试手动运行以下命令排查：
        echo         "%OLLAMA_EXE%" list
        pause
        exit /b 1
    )
)
echo         模型 qwen2.5:1.5b 已验证可用！

:: =========================================================================
:: 完成
:: =========================================================================
echo.
echo  ======================================================================
echo    部署成功！
echo  ======================================================================
echo.
echo    Ollama 后台服务 : 已启动 （端口 11434）
echo    开机自启       : 已配置 （登录后自动隐藏运行）
echo    托盘图标       : 已屏蔽 （医生无感）
echo    离线模型       : qwen2.5:1.5b (~986MB)
echo.
echo    TCM 纠错模块已就绪，可直接运行 error_correction.py。
echo.
echo  ======================================================================
echo.
echo  提示：如需测试，请打开终端并运行：
echo        python error_correction.py
echo.
pause
exit /b 0
