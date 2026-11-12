"""Texts of a GEN file as the raters read them (blind_rate.py's cut, 460 characters)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from blind_rate import cut, read_gen  # noqa: E402


def cut_texts(path, n=460):
    return {i: cut(t, n) for i, t in read_gen(path).items()}
