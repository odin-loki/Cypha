# CyphaStudio GUI threading

## Training worker

`MainWindow.TrainingWorker` is a `QThread`. Its `run()` method:

- Calls `Trainer.fit()` and `Trainer.evaluate()`.
- Uses a `TrainerCallback` that only calls `SignalBus` emit helpers (`emit_training_step`, `emit_training_evaluated`, `emit_training_finished`, `emit_error`).

Qt signals emitted from a secondary thread are **queued** to the GUI thread, so slots that update widgets (training curves, status bar, log dock) run on the main thread.

## Rules for new code

1. Do **not** read or write Qt widgets from inside `TrainingWorker.run()` or from trainer callbacks except via **signals**.
2. Prefer `SignalBus` for cross-thread status; keep `QMessageBox` and other UI on slots connected to those signals (main thread).

## REST / headless

The FastAPI app runs inference on the asyncio thread pool’s worker by default (sync handlers). That is separate from the Qt GUI path; do not mix Qt calls into API handlers.
