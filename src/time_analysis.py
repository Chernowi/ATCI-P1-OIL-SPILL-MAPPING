import cProfile
import pstats
import io
# Your program's main function or entry point
from main import main

profiler = cProfile.Profile()
profiler.enable()

# --- Run your code ---
main("default_mapping")
# ---------------------

profiler.disable()

s = io.StringIO()
# Sort stats by cumulative time spent in function
stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')

stats.sort_stats('tottime').print_stats(20)

print(s.getvalue())

# Optionally save to a file for later analysis
profiler.dump_stats('my_program.prof')