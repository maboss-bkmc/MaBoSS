echo off

if not exist tmp\ mkdir tmp\

set /A FAIL=0

IF "%MABOSS_128n%"=="" set MABOSS_128n=..\pub\MaBoSS_128n.exe

echo "Test: Ewing one thread"

%MABOSS_128n% ewing\ewing_full.bnd -c ewing\ewing.cfg -c ewing\ewing_runcfg-thread_1.cfg -o tmp\ewing_thread_1
call:check Run

python compare_probtrajs.py ewing\refer\ewing_thread_1_probtraj.csv tmp\ewing_thread_1_probtraj.csv --exact
call:check ProbTraj

python compare_statdist.py ewing\refer\ewing_thread_1_statdist.csv tmp\ewing_thread_1_statdist.csv --exact 
call:check StatDist

python compare_fixpoints.py ewing\refer\ewing_thread_1_fp.csv tmp\ewing_thread_1_fp.csv
call:check Fixpoints

echo "Test: Ewing 6 threads"

%MABOSS_128n% ewing\ewing_full.bnd -c ewing\ewing.cfg -c ewing\ewing_runcfg-thread_6.cfg -o tmp\ewing_thread_6
call:check Run

python compare_probtrajs.py ewing\refer\ewing_thread_6_probtraj.csv tmp\ewing_thread_6_probtraj.csv --exact
call:check ProbTraj

python compare_statdist.py ewing\refer\ewing_thread_6_statdist.csv tmp\ewing_thread_6_statdist.csv --exact 
call:check StatDist

python compare_fixpoints.py ewing\refer\ewing_thread_6_fp.csv tmp\ewing_thread_6_fp.csv
call:check Fixpoints

echo "Test: checking differences between one and 6 threads results"
python compare_probtrajs.py tmp\ewing_thread_1_probtraj.csv tmp\ewing_thread_6_probtraj.csv
call:check ProbTraj

python compare_statdist.py tmp\ewing_thread_1_statdist.csv tmp\ewing_thread_6_statdist.csv --exact
call:check StatDist


exit /b %FAIL%


:check 
if %ERRORLEVEL% EQU 0 (
    echo %~1 : OK
) else (
    echo %~1 : ERR
    set /A FAIL=1
)
exit /b 