@echo off
setlocal enabledelayedexpansion
title TCM ASR 离线模型准备工具

:: ============================================================================
:: prepare_offline_model.bat — 在开发机上运行一次，生成离线部署所需的模型文件
:: ============================================================================

echo.
echo  ================================================================
echo    TCM ASR - 离线模型准备工具
echo  ================================================================
echo.
echo  此脚本将把当前已下载的 qwen2.5:1.5b 模型文件
echo  导出到 .\ollama_model_offline\ 目录，供离线部署使用。
echo.

set "SOURCE_MODELS=%USERPROFILE%\.ollama\models"
set "SOURCE_CONFIG=%LOCALAPPDATA%\Ollama"
set "TARGET_DIR=%~dp0ollama_model_offline"

:: ---- 检查源模型是否存在 ----
if not exist "%SOURCE_MODELS%\blobs\" (
    echo  [ERROR] 未找到模型文件！请先运行: ollama pull qwen2.5:1.5b
    pause
    exit /b 1
)

:: ---- 创建目标目录 ----
if exist "%TARGET_DIR%" (
    echo  [INFO] 目标目录已存在，正在清理...
    rd /s /q "%TARGET_DIR%"
)
mkdir "%TARGET_DIR%\models\blobs"
mkdir "%TARGET_DIR%\models\manifests"
mkdir "%TARGET_DIR%\config"

:: ---- 拷贝模型 blobs ----
echo  [1/3] 拷贝模型权重文件 (blobs) ...
xcopy /e /i /q /y "%SOURCE_MODELS%\blobs\*" "%TARGET_DIR%\models\blobs\"
if errorlevel 1 (
    echo  [ERROR] 拷贝 blobs 失败！
    pause
    exit /b 1
)
echo         完成。

:: ---- 拷贝模型清单 ----
echo  [2/3] 拷贝模型清单 (manifests) ...
xcopy /e /i /q /y "%SOURCE_MODELS%\manifests\*" "%TARGET_DIR%\models\manifests\"
if errorlevel 1 (
    echo  [ERROR] 拷贝 manifests 失败！
    pause
    exit /b 1
)
echo         完成。

:: ---- 拷贝 Ollama 配置数据库（包含模型注册信息） ----
echo  [3/3] 拷贝 Ollama 配置数据 ...
if exist "%SOURCE_CONFIG%\db.sqlite"  copy /y "%SOURCE_CONFIG%\db.sqlite"  "%TARGET_DIR%\config\" >nul 2>&1
if exist "%SOURCE_CONFIG%\db.sqlite-shm" copy /y "%SOURCE_CONFIG%\db.sqlite-shm" "%TARGET_DIR%\config\" >nul 2>&1
if exist "%SOURCE_CONFIG%\db.sqlite-wal" copy /y "%SOURCE_CONFIG%\db.sqlite-wal" "%TARGET_DIR%\config\" >nul 2>&1
if exist "%USERPROFILE%\.ollama\id_ed25519"     copy /y "%USERPROFILE%\.ollama\id_ed25519"     "%TARGET_DIR%\config\" >nul 2>&1
if exist "%USERPROFILE%\.ollama\id_ed25519.pub" copy /y "%USERPROFILE%\.ollama\id_ed25519.pub" "%TARGET_DIR%\config\" >nul 2>&1
echo         完成。

:: ---- 汇总 ----
echo.
echo  ================================================================
echo    离线模型包准备完毕！
echo  ================================================================
echo.
echo  输出目录: %TARGET_DIR%
for /f "tokens=3" %%a in ('dir /s /-c "%TARGET_DIR%" ^| find "个文件"') do set FILE_COUNT=%%a
dir "%TARGET_DIR%" /s | find "文件" 2>nul
echo.
echo  部署时需要此目录 + OllamaSetup.exe + hospital_install.bat
echo  一并拷贝到 U 盘即可。
echo.
pause
