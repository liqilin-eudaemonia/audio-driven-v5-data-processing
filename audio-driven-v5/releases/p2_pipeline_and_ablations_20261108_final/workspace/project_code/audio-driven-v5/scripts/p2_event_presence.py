#!/usr/bin/env python3
"""Explicit independent stage for clip-level seven-class event evaluation."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_event_presence.cli import main

if __name__=='__main__':main()
