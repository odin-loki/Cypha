#!/usr/bin/env python3
"""
CyphaStudio — entry point.

Usage:
    python main.py                              # launch GUI
    python main.py --model iris-clf --ver 1.0.0 # load model on start
    python main.py --dataset iris               # load sklearn dataset
    python main.py --headless                   # REST API only, no GUI
    python main.py --headless --port 7749       # specify port
    python main.py --headless --regression-head path/to/regression_head.json
    python main.py --train --dataset iris       # train then launch GUI
"""
import argparse
import os
import sys

# Add parent directory so Cypha.py is importable
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    from cypha_studio.env_config import api_default_host, api_default_port

    p = argparse.ArgumentParser(description="CyphaStudio")
    p.add_argument('--model',    type=str, help='Model name to load on start')
    p.add_argument('--version',  type=str, default='latest', help='Model version')
    p.add_argument('--dataset',  type=str, help='Dataset to load (csv path or sklearn name)')
    p.add_argument('--headless', action='store_true', help='REST API only, no GUI')
    p.add_argument(
        '--port',
        type=int,
        default=None,
        help='REST API port (default: CYPHA_API_PORT or 7749)',
    )
    p.add_argument(
        '--host',
        type=str,
        default=None,
        help='REST API host (default: CYPHA_API_HOST or 127.0.0.1)',
    )
    p.add_argument('--train',    action='store_true', help='Start training immediately')
    p.add_argument(
        '--regression-head',
        type=str,
        default='',
        metavar='PATH',
        help='regression_head.json for /predict MoE (overrides CYPHA_REGRESSION_HEAD when set)',
    )
    args = p.parse_args()
    if args.host is None:
        args.host = api_default_host()
    if args.port is None:
        args.port = api_default_port()
    return args


def run_headless(args):
    """Start the REST API without a GUI."""
    from cypha_studio.core.registry  import ModelRegistry
    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.env_config     import registry_root
    from cypha_studio.server.api     import start_server

    registry = ModelRegistry(registry_root())
    engine   = None
    session  = None

    if args.model:
        try:
            model, pre, card = registry.load(args.model, args.version)
            engine  = InferenceEngine(model, pre)
            session = InferenceSession(engine)
            print(f"[CyphaStudio] Model loaded: {card.name} v{card.version}")
        except Exception as e:
            print(f"[CyphaStudio] Failed to load model: {e}")

    print(f"[CyphaStudio] Starting REST API on {args.host}:{args.port}")
    reg_head = args.regression_head.strip() or None
    start_server(
        engine=engine, registry=registry, session=session,
        host=args.host, port=args.port,
        regression_head_path=reg_head,
    )


def run_gui(args):
    """Launch the Qt GUI application."""
    from PySide6.QtWidgets import QApplication
    from cypha_studio.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("CyphaStudio")
    app.setOrganizationName("Cypha")

    window = MainWindow()

    # Load model if specified
    if args.model:
        window._load_model_by_name(args.model, args.version)

    # Load dataset if specified
    if args.dataset:
        if os.path.exists(args.dataset):
            window.dataset_widget.load_file(args.dataset)
        else:
            # Try as sklearn dataset name
            window.dataset_widget._load_sklearn(args.dataset)

    # Auto-start training if requested
    if args.train and args.dataset:
        from PySide6.QtCore import QTimer
        QTimer.singleShot(500, window._on_start_train)

    window.show()
    sys.exit(app.exec())


def main():
    args = parse_args()
    if args.headless:
        run_headless(args)
    else:
        run_gui(args)


if __name__ == '__main__':
    main()
