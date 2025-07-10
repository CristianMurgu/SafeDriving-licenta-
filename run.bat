@echo off
cd /d "C:\Users\admin\PycharmProjects\SafeDrivingProject"
call .venv\Scripts\activate
start "" /B .venv\Scripts\pythonw.exe main.py

:: :kill_switch
:: choice /c k /n >nul
:: if errorlevel 1 (
   :: taskkill /f /im pythonw.exe
   :: exit
:: )
:: goto kill_switch