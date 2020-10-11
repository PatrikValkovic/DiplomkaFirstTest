import argparse
import tensorflow as tf
import numpy as np
from progressbar import progressbar
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument("--gpu", type=bool, default=True, help="Use GPU")
argparser.add_argument("--population", type=int, default=1000, help="Size of the population")
argparser.add_argument("--iterations", type=int, default=100_000, help="Number of iterations")
argparser.add_argument("--dimension", type=int, default=10, help="Dimension of the problem")
argparser.add_argument("--kill_factor", type=int, default=0.6, help="Dimension of the problem")
argparser.add_argument("--CR", type=float, default=0.4, help="Differential evolution CR term")
argparser.add_argument("--F", type=float, default=0.8, help="Differential evolution F term")
args, _ = argparser.parse_known_args()

tf.config.set_visible_devices(tf.config.get_visible_devices("GPU") if args.gpu else [], 'GPU')

def create_fn1():
    xopt = tf.random.uniform(shape=(args.dimension,), minval=-10, maxval=10)
    fopt = np.random.rand() * 200.0 - 100.0
    @tf.function
    def fn1(members: tf.Tensor):
        dot = members - xopt[tf.newaxis,:]
        value = tf.reduce_sum(dot * dot, axis=1)
        return (value + fopt, value)
    return fn1

def create_fn1mapfn():
    xopt = tf.random.uniform(shape=(args.dimension,), minval=-10, maxval=10)
    fopt = np.random.rand() * 200.0 - 100.0
    @tf.function
    def axis_fn(example: tf.Tensor):
        minus = example - xopt
        dot = tf.reduce_sum(tf.multiply(minus, minus), axis=0)
        return dot
    @tf.function
    def fn1(members: tf.Tensor):
        value = tf.map_fn(axis_fn, members, parallel_iterations=members.shape[0])
        return (value + fopt, value)
    return fn1

fn = create_fn1()

@tf.function
def one_iteration(population):
    fitneses, real_fitness = fn(population)  # type: (tf.Tensor, tf.Tensor)
    num_parents = int(args.population * args.kill_factor)
    parents_tournament_indices = tf.random.uniform(shape=(2, num_parents), minval=0, maxval=args.population, dtype=tf.int32)
    comparison = tf.gather(fitneses, parents_tournament_indices[0], axis=0) < tf.gather(fitneses, parents_tournament_indices[1], axis=0)
    better = tf.concat([
        tf.boolean_mask(parents_tournament_indices[0], comparison, axis=0),
        tf.boolean_mask(parents_tournament_indices[1], tf.math.logical_not(comparison), axis=0)
    ], axis=0)
    parents = tf.gather(population, better, axis=0)

    # create children
    num_children = args.population - num_parents
    picked_parents = tf.random.uniform(shape=(4, num_children), minval=0, maxval=num_parents, dtype=tf.int32)
    crossover_sample = tf.random.uniform(shape=(num_children, args.dimension), minval=0, maxval=1) > args.CR
    mutated = tf.gather(parents, picked_parents[0], axis=0) + args.F * (tf.gather(parents, picked_parents[1], axis=0) - tf.gather(parents, picked_parents[2]))
    #mutated[crossover_sample] = parents[picked_parents[3]][crossover_sample]
    indices = tf.where(crossover_sample)
    mutated = tf.tensor_scatter_nd_update(mutated, indices, 
        tf.boolean_mask(tf.gather(parents, picked_parents[3], axis=0), crossover_sample, axis=0)
    )
    mutated = tf.clip_by_value(mutated, -10, 10)

    population = tf.concat([parents, mutated], axis=0)
    return population, tf.reduce_mean(real_fitness, axis=0)


relative_progress = np.zeros(shape=(args.iterations,))
population = tf.random.uniform(shape=(args.population, args.dimension), minval=-10, maxval=10)
for iter in progressbar(range(args.iterations)):
    population, progress = one_iteration(population)
    relative_progress[iter] = progress


plt.plot(relative_progress)
plt.yscale('log')
plt.show()
