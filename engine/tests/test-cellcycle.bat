@echo off

echo "Test: Cell Cycle 1 threads"

if not exist tmp\ mkdir tmp\
IF "%MABOSS%"=="" set MABOSS=..\pub\MaBoSS.exe

set /A FAIL=0
%MABOSS% cellcycle\cellcycle.bnd -c cellcycle\cellcycle_runcfg.cfg -c cellcycle\cellcycle_runcfg-thread_1.cfg -o tmp\Cell_cycle_thread_1 > nul 2>&1
call:check Run

python compare_probtrajs.py cellcycle\refer\Cell_cycle_thread_1_probtraj.csv tmp\Cell_cycle_thread_1_probtraj.csv --exact
call:check ProbTraj

python compare_statdist.py cellcycle\refer\Cell_cycle_thread_1_statdist.csv tmp\Cell_cycle_thread_1_statdist.csv --exact 
call:check StatDist
  
python compare_fixpoints.py cellcycle\refer\Cell_cycle_thread_1_fp.csv tmp\Cell_cycle_thread_1_fp.csv
call:check Fixpoints

echo "Test: Cell Cycle 6 threads"
%MABOSS% cellcycle\cellcycle.bnd -c cellcycle\cellcycle_runcfg.cfg -c cellcycle\cellcycle_runcfg-thread_6.cfg -o tmp\Cell_cycle_thread_6 > nul 2>&1
call:check Run

python compare_probtrajs.py cellcycle\refer\Cell_cycle_thread_6_probtraj.csv tmp\Cell_cycle_thread_6_probtraj.csv --exact
call:check ProbTraj

echo "Test: checking differences between one and 6 threads results"
python compare_statdist.py cellcycle\refer\Cell_cycle_thread_6_statdist.csv tmp\Cell_cycle_thread_6_statdist.csv --exact 
call:check StatDist

python compare_fixpoints.py cellcycle\refer\Cell_cycle_thread_6_fp.csv tmp\Cell_cycle_thread_6_fp.csv
call:check Fixpoints


exit /b %FAIL%

:check 
if %ERRORLEVEL% EQU 0 (
    echo %~1 : OK
) else (
    echo %~1 : ERR
    set /A FAIL=1
)
exit /b 