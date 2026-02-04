import sys
import dis
import os
from collections import Counter

class TraceLogger:
    def __init__(self, target_filename, core_start_line=None):
        """
        Args:
            target_filename (str): Absolute path to the file to trace (solution).
            core_start_line (int): 1-based line number to start counting opcodes from. If None, call find_marker().
        """
        self.target_filename = os.path.abspath(target_filename)
        self.core_start_line = core_start_line
        self.opcodes = Counter()
        self.active = False
        
        # Opcode map for faster lookups (optional optimization, but dis.opname is array lookup)
        self.opname = dis.opname

    def trace_func(self, frame, event, arg):
        # Unify filename check
        # Note: frame.f_code.co_filename might be relative or absolute.
        # We must normalize it to match self.target_filename (which is absolute).
        current_abs_path = os.path.abspath(frame.f_code.co_filename)
        
        if current_abs_path == self.target_filename:
             frame.f_trace_opcodes = True
            
        if event == 'opcode':
            # We only count if we are in the target file
            if current_abs_path == self.target_filename:
                # Check line number
                if frame.f_lineno is not None and frame.f_lineno >= self.core_start_line:
                    try:
                        # f_lasti is index into bytecode string
                        lasti = frame.f_lasti
                        code = frame.f_code.co_code
                        op = code[lasti]
                        
                        # In Python 3.11+, opcodes are adaptive, might need processing.
                        # But `dis.opname[op]` works for standard opcodes.
                        name = self.opname[op]
                        self.opcodes[name] += 1
                    except Exception:
                        pass
        return self.trace_func


    def find_marker(self, marker="# === TRACE_BEGIN ==="):
        """
        Scans the target file to find the marker line number (1-based).
        Updates self.core_start_line.
        """
        try:
            with open(self.target_filename, "r") as f:
                for i, line in enumerate(f):
                    if marker in line:
                        # Marker is on line i+1. Start counting AFTER marker.
                        self.core_start_line = i + 2
                        return
        except Exception:
            pass
        # If not found, default to 1 or keep existing
        if self.core_start_line is None:
             self.core_start_line = 1

    def start(self):
        if self.core_start_line is None:
             self.find_marker()
        self.active = True
        sys.settrace(self.trace_func)

    def stop(self):
        self.active = False
        sys.settrace(None)
        
    def get_counts(self):
        return dict(self.opcodes)
