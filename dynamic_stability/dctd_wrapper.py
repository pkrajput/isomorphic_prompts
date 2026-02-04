import sys
import runpy
import json
import os
import traceback

# Usage: python dctd_wrapper.py <target_file> <trace_out_file>
# The script passes stdin/stdout through to the target.

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dctd_wrapper.py <target_file> <trace_out_file>", file=sys.stderr)
        sys.exit(1)
        
    target_file = sys.argv[1]
    trace_out = sys.argv[2]
    
    # Add dynamic_stability to path to find trace_runtime if needed?
    # Assume trace_runtime is in the same dir as wrapper or installed?
    # We will invoke wrapper from calculate_dctd with proper PYTHONPATH.
    
    try:
        from trace_runtime import TraceLogger
    except ImportError:
        # Fallback: assume sibling
        sys.path.append(os.path.dirname(__file__))
        from trace_runtime import TraceLogger
        
    tracer = TraceLogger(target_file)
    # find marker explicitly
    tracer.find_marker()
    
    # We want to capture opcodes ONLY from the target file execution.
    tracer.start()
    
    try:
        # Run the file
        # We use run_path to execute as script
        # This allows it to read stdin if it wants.
        runpy.run_path(target_file, run_name="__main__")
        
    except SystemExit as e:
        # Propagate exit code?
        if e.code is not None and e.code != 0:
            # We don't want wrapper to fail just because solution failed (unless it crashed)
            # But for "traced run failed" detection, we might want to propagate.
             print(f"Wrapper: SystemExit {e.code}", file=sys.stderr)
             # But we MUST save trace first.
    except Exception as e:
        print(f"Wrapper: Exception {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1) # Signal failure
        
    finally:
        tracer.stop()
        
        # Write outputs
        try:
            with open(trace_out, "w") as f:
                counts = tracer.get_counts()
                json.dump(counts, f)
        except Exception as e:
            print(f"Wrapper: Failed to write trace: {e}", file=sys.stderr)

