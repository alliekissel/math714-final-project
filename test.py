import numpy as np
import pickle
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
from single_pendulum import single_pendulum

dir_path = './output/oa/'

to_xyz = 'xyz'
pretty_form = {'rp': 'rp', 'rA': 'rA'}

def run_model(args):
    form, model_fn, num_bodies = args

    steps = 1e-4

    pretty_name = ' '.join([word.capitalize() for word in model_fn.__name__.split('_')])
    
    try:
        pos, vel, acc, _, grid = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(1e-11/steps**2),'--end_time', str(1.0), '--step_size', str(steps)])

    except RuntimeError:
        print('{}-{}, step: {}, tol: {} failed to converge'.format(form, pretty_name, str(steps), str(1e-11/steps**2)))
    except ValueError:
        print('{}-{}, step: {}, tol: {} raised value error'.format(form, pretty_name, str(steps), str(1e-11/steps**2)))


    print('Completed {} {} Analysis'.format(pretty_name, pretty_form[form]))

    # plot 
    plt.plot(grid, pos[0,2,:])
    plt.show()

tasks = []

for model_fn in [single_pendulum]:
    num_bodies = 1 if model_fn.__name__ == 'single_pendulum' else 3

    for form in ['rp']:
        tasks.append((form, model_fn, num_bodies))

for task in tasks:
    run_model(task)
