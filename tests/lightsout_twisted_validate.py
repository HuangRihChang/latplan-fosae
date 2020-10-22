#!/usr/bin/env python3

import numpy as np

import latplan
import latplan.puzzles.lightsout_twisted as p

import itertools
c = np.array([ c for c in itertools.islice(p.generate_configs(4), 10000) ])

from functools import partial

from latplan.util.timer import Timer

with Timer("************************* states on cpu ***************************"):
    s = p.generate_cpu(c)

with Timer("************************* states on gpu, batch=100  ***************************"):
    s = p.generate_gpu(c, batch_size=100)

with Timer("************************* states on gpu, batch=1000 ***************************"):
    s = p.generate_gpu(c, batch_size=1000)

with Timer("************************* states on gpu2, batch=100  ***************************"):
    s = p.generate_gpu2(c, batch_size=100)

with Timer("************************* states on gpu2, batch=1000 ***************************"):
    s = p.generate_gpu2(c, batch_size=1000)

for i in range(1,5):
    p.threshold = i*0.01
    with Timer("************************* validate_states with threshold = {} ***************************".format(p.threshold))):
        results = p.validate_states(s,batch_size=1000)
    print("results:", np.all(results), "(should be True)")
    print("how many invalid? : ", len(results)-np.count_nonzero(results), "/", len(results))

with Timer("************************* validate_states with noise ***************************"):
    p.threshold = 0.04
    print("results:", np.all(p.validate_states(np.clip(s+np.random.normal(0,0.1,s.shape),0,1))), "(should be True)")

with Timer("************************* to_configs on gpu, batch=100 ***************************"):
    p.to_configs(s,batch_size=100)
    

with Timer("************************* to_configs on gpu, batch=1000 ***************************"):
    p.to_configs(s,batch_size=1000)

from latplan.util import bce, mae
_c = p.to_configs(s,batch_size=1000).round().astype(int)
print("sum(abs(config - to_configs(generate(config)))) =", np.sum(np.abs(c - _c)))
print("MAE(config, to_configs(generate(config))) =", mae(c, _c))
print("BCE(config, to_configs(generate(config))) =", bce(c, _c))
for i in range(120,125):
    print("original     :",c[i])
    print("reconstructed:",_c[i])
    

c = c[:10]

with Timer("************************* transitions_old ***************************"):
    transitions = p.transitions_old(4,configs=c)

with Timer("************************* transitions ***************************"):
    transitions = p.transitions(4,configs=c)

with Timer("************************* transitions one_per_state ***************************"):
    transitions = p.transitions(4,configs=c,one_per_state=True)

with Timer("************************* validate_transitions_cpu ***************************"):
    results = p.validate_transitions(transitions,batch_size=1000)

print("all transitions valid?:",np.all(results))
print("if not, how many invalid?:",len(results)-np.count_nonzero(results), "/", len(results))

