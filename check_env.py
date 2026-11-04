"""Run once: python check_env.py --instance-seed 5 --story A --variant 1  (add --cap 108 for T11).
Compare the printed values with the check values of your task file."""
import argparse, sys
print('Python', sys.version.split()[0])
ok = True
for m in ('numpy', 'matplotlib'):
    try:
        mod = __import__(m); print(m, mod.__version__)
    except Exception as e:
        ok = False; print('MISSING', m, e)
try:
    import checkpoint4_generator_PUBLIC as gen
except Exception as e:
    print('Put checkpoint4_generator_PUBLIC.py next to this file:', e); sys.exit(1)
ap = argparse.ArgumentParser()
ap.add_argument('--instance-seed', type=int, required=True); ap.add_argument('--story', required=True)
ap.add_argument('--variant', type=int, required=True); ap.add_argument('--cap', type=int, default=None)
a = ap.parse_args()
d = gen.cp34_instance(a.instance_seed, a.story, a.variant, cap=a.cap)
if a.story == 'A':
    print('N', d['N'], 'V', d['V'], 'Sum demand', int(sum(d['demand'])), 'Cap', d.get('Cap'))
    print('sum of all distances (3 decimals): %.3f' % d['dist'].sum())
else:
    print('S', d['S'], 'M', d['M'], 'cpu_cap', int(sum(d['cpu_cap'])), 'mem_cap', int(sum(d['mem_cap'])),
          'cpu_req', int(sum(d['cpu_req'])), 'mem_req', int(sum(d['mem_req'])))
print('Environment OK' if ok else 'Install the missing packages')
