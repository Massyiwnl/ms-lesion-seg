@echo off
REM Esegue in sequenza tutti gli esperimenti dell'ablation + valutazione sul test set.
REM Uso:  scripts\run_ablation.bat
REM Durata stimata: ~6 ore (5 run di training da ~55 min + valutazioni).
REM I run gia' completati NON vengono saltati: commenta le righe che non ti servono.

setlocal
set EP=--override train.epochs=60 data.samples_per_epoch=6000 train.batch_size=16

echo === E1: multimodale FLAIR+T1+T2 ===
call :run python scripts/02_train.py --config configs/exp_unet_multimodal.yaml %EP%
call :run python scripts/03_evaluate.py --run runs/unet_multimodal --split test

echo === E2: Attention U-Net ===
call :run python scripts/02_train.py --config configs/exp_attunet_flair.yaml %EP%
call :run python scripts/03_evaluate.py --run runs/attunet_flair --split test

echo === E3: deep supervision ===
call :run python scripts/02_train.py --config configs/exp_unet_flair_ds.yaml %EP%
call :run python scripts/03_evaluate.py --run runs/unet_flair_ds --split test

echo === E4: loss Dice+Focal ===
call :run python scripts/02_train.py --config configs/exp_unet_flair_focal.yaml %EP%
call :run python scripts/03_evaluate.py --run runs/unet_flair_focal --split test

echo === E5: loss Tversky ===
call :run python scripts/02_train.py --config configs/exp_unet_flair_tversky.yaml %EP%
call :run python scripts/03_evaluate.py --run runs/unet_flair_tversky --split test

echo === TABELLA COMPARATIVA ===
call :run python scripts/04_compare.py --runs-dir runs --split test --baseline unet_flair

echo.
echo Fatto. Controlla runs\_comparison\comparison_test.md
endlocal
goto :eof

:run
%*
if errorlevel 1 (
  echo.
  echo [ERRORE] Comando fallito: %*
  echo Interrompo la sequenza.
  endlocal
  exit /b 1
)
goto :eof
