@echo off

echo "Test: Ensemble"

if not exist tmp\ mkdir tmp\
IF "%MABOSS%"=="" set MABOSS=..\pub\MaBoSS.exe

set /A FAIL=0

%MABOSS% --ensemble --save-individual -c ensemble\ensemble.cfg -o tmp\res ensemble\invasion\Invasion_0.bnet ensemble\invasion\Invasion_200.bnet ensemble\invasion\Invasion_400.bnet ensemble\invasion\Invasion_600.bnet ensemble\invasion\Invasion_800.bnet  ensemble\invasion\Invasion_1000.bnet
call:check Run

python compare_probtrajs.py ensemble\refer\res_probtraj.csv tmp\res_probtraj.csv --exact
call:check "projtraj"
python compare_probtrajs.py ensemble\refer\res_model_0_probtraj.csv tmp\res_model_0_probtraj.csv --exact
call:check "projtraj_model_0"
python compare_probtrajs.py ensemble\refer\res_model_1_probtraj.csv tmp\res_model_1_probtraj.csv --exact
call:check "projtraj_model_1"
python compare_probtrajs.py ensemble\refer\res_model_2_probtraj.csv tmp\res_model_2_probtraj.csv --exact
call:check "projtraj_model_2"
python compare_probtrajs.py ensemble\refer\res_model_3_probtraj.csv tmp\res_model_3_probtraj.csv --exact
call:check "projtraj_model_3"
python compare_probtrajs.py ensemble\refer\res_model_4_probtraj.csv tmp\res_model_4_probtraj.csv --exact
call:check "projtraj_model_4"
python compare_probtrajs.py ensemble\refer\res_model_5_probtraj.csv tmp\res_model_5_probtraj.csv --exact
call:check "projtraj_model_5"

python compare_fixpoints.py ensemble\refer\res_fp.csv tmp\res_fp.csv
call:check "fixpoints"

python compare_fixpoints.py ensemble\refer\res_model_0_fp.csv tmp\res_model_0_fp.csv
call:check "fixpoints_model_0"
python compare_fixpoints.py ensemble\refer\res_model_1_fp.csv tmp\res_model_1_fp.csv
call:check "fixpoints_model_1"
python compare_fixpoints.py ensemble\refer\res_model_2_fp.csv tmp\res_model_2_fp.csv
call:check "fixpoints_model_2"
python compare_fixpoints.py ensemble\refer\res_model_3_fp.csv tmp\res_model_3_fp.csv
call:check "fixpoints_model_3"
python compare_fixpoints.py ensemble\refer\res_model_4_fp.csv tmp\res_model_4_fp.csv
call:check "fixpoints_model_4"
python compare_fixpoints.py ensemble\refer\res_model_5_fp.csv tmp\res_model_5_fp.csv
call:check "fixpoints_model_5"

exit /b %FAIL%

:check 
if %ERRORLEVEL% EQU 0 (
    echo %~1 : OK
) else (
    echo %~1 : ERR
    set /A FAIL=1
)
exit /b 