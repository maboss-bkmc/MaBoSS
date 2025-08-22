echo off

set MABOSS="%PREFIX%\bin\MaBoSS.exe"
set POPMABOSS="%PREFIX%\bin\PopMaBoSS.exe"
set MABOSS_128n="%PREFIX%\bin\MaBoSS_128n.exe"

cd %PREFIX%\tests\maboss

set /A FAIL=0
call .\test-cellcycle.bat
call:check

call .\test-ewing.bat
call:check

call .\test-ensemble.bat
call:check

call .\test-popmaboss.bat
call:check

exit /b %FAIL%



:check 
if %ERRORLEVEL% NEQ 0 (
    set /A FAIL=1
)
exit /b 