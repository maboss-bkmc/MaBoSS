echo off

if not exist tmp\ mkdir tmp\

set /A FAIL=0
IF "%POPMABOSS%"=="" set POPMABOSS=..\pub\PopMaBoSS.exe

echo "Test: PopMaBoSS"

%POPMABOSS% -c popmaboss\Fork.cfg -o tmp\res_fork popmaboss\Fork.bnd > nul 2>&1
call:check Run

python compare_probtrajs.py popmaboss\refer\res_fork_pop_probtraj.csv tmp\res_fork_pop_probtraj.csv 1e-2 1e-4
call:check Probtraj

%POPMABOSS% -c popmaboss\Fork.pcfg -o tmp\res_fork popmaboss\Fork.bnd > nul 2>&1
call:check Run

python compare_probtrajs.py popmaboss\refer\res_fork_pop_probtraj.csv tmp\res_fork_pop_probtraj.csv 1e-2 1e-4
call:check Probtraj

%POPMABOSS% -c popmaboss\Log_Growth.cfg -o tmp\res_log_growth popmaboss\Log_Growth.pbnd > nul 2>&1
call:check Run

python compare_probtrajs.py popmaboss\refer\res_log_growth_pop_probtraj.csv tmp\res_log_growth_pop_probtraj.csv --exact
call:check Probtraj

%POPMABOSS% -c popmaboss\Assymetric.cfg -o tmp\res_assymetric popmaboss\Assymetric.pbnd > nul 2>&1
call:check Run

python compare_probtrajs.py popmaboss\refer\res_assymetric_pop_probtraj.csv tmp\res_assymetric_pop_probtraj.csv 5e-2 5e-2
call:check Probtraj

%POPMABOSS% -c popmaboss\ICD_phenomenological_TDC_ratio.cfg -o tmp\res_icd popmaboss\ICD_phenomenologicalPM.pbnd  > nul 2>&1
call:check Run

python compare_probtrajs.py popmaboss\refer\res_icd_custom_pop_probtraj.csv tmp\res_icd_custom_pop_probtraj.csv 5e-2 5e-2
call:check Probtraj

exit /b %FAIL%


:check 
if %ERRORLEVEL% EQU 0 (
    echo %~1 : OK
) else (
    echo %~1 : ERR
    set /A FAIL=1
)
exit /b 