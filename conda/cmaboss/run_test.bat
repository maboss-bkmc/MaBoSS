echo off

cd %PREFIX%\tests

set /A FAIL=0

python -m unittest test
call:check
python -m unittest test_128n
call:check

cd ../..


exit /b %FAIL%



:check 
if %ERRORLEVEL% NEQ 0 (
    set /A FAIL=1
)
exit /b 