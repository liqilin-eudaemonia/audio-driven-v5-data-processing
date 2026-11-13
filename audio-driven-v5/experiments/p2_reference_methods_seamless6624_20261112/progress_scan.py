"""Read append-only completion receipts without probing every future sample path."""
import os
from pathlib import Path

class ReceiptScan:
    def __init__(self,allowed):self.allowed=set(allowed);self.known={}
    def ids(self,parent,marker='complete.json'):
        parent=Path(parent);known=self.known.setdefault((str(parent),marker),set())
        try:
            with os.scandir(parent) as entries:
                for entry in entries:
                    if entry.name not in self.allowed or entry.name in known:continue
                    if (Path(entry.path)/marker).is_file():known.add(entry.name)
        except FileNotFoundError:pass
        return known.copy()
