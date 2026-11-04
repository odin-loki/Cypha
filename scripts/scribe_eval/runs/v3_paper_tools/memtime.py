# Wall time and peak RSS of a command (the container has no /usr/bin/time). Author: Odin Loch
import resource, subprocess, sys, time
t = time.time()
r = subprocess.run(sys.argv[1:])
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
sys.stderr.write(f"MEMTIME wall_s={time.time()-t:.2f} user_s={ru.ru_utime:.2f} sys_s={ru.ru_stime:.2f} maxrss_mb={ru.ru_maxrss/1024:.1f} rc={r.returncode}\n")
sys.exit(r.returncode)
