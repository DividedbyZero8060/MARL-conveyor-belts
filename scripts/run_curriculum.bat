@echo off
rem Curriculum chain orchestrator (Windows).
rem Usage: run_curriculum.bat <seed> [stage]
setlocal
set SEED=%1
if "%SEED%"=="" set SEED=42
set START_STAGE=%2
if "%START_STAGE%"=="" set START_STAGE=1

set STAGE1_ID=curriculum_s%SEED%_stage1
set STAGE2_ID=curriculum_s%SEED%_stage2
set STAGE3_ID=curriculum_s%SEED%_stage3

if %START_STAGE% LEQ 1 (
  echo.
  echo ^>^>^> STAGE 1: Single-type task, 200k steps.
  echo ^>^>^> Set PackageSpawner._packageTypes = [SmallParcel] in Inspector.
  pause
  mlagents-learn Assets/Config/SortingAgent_stage1.yaml --run-id=%STAGE1_ID% --seed=%SEED% --results-dir=results --no-graphics --force
  if errorlevel 1 goto :error
)

if %START_STAGE% LEQ 2 (
  echo.
  echo ^>^>^> STAGE 2: Two-type task, 400k steps, warm-started from Stage 1.
  echo ^>^>^> Set PackageSpawner._packageTypes = [SmallParcel, MediumBox] in Inspector.
  pause
  mlagents-learn Assets/Config/SortingAgent_stage2.yaml --run-id=%STAGE2_ID% --initialize-from=%STAGE1_ID% --seed=%SEED% --results-dir=results --no-graphics --force
  if errorlevel 1 goto :error
)

if %START_STAGE% LEQ 3 (
  echo.
  echo ^>^>^> STAGE 3: Three-type task, 1M steps, warm-started from Stage 2.
  echo ^>^>^> Set PackageSpawner._packageTypes = [SmallParcel, MediumBox, LargeCrate] in Inspector.
  pause
  mlagents-learn Assets/Config/SortingAgent_stage3.yaml --run-id=%STAGE3_ID% --initialize-from=%STAGE2_ID% --seed=%SEED% --results-dir=results --no-graphics --force
  if errorlevel 1 goto :error
)

echo.
echo Curriculum chain finished. Seed %SEED%.
echo TensorBoard: tensorboard --logdir results
exit /b 0

:error
echo ERROR: stage failed.
exit /b 1