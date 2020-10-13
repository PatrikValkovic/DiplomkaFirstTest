import argparse
import tensorflow as tf
import numpy as np
from progressbar import progressbar
import matplotlib.pyplot as plt
from timeit import default_timer as timer

argparser = argparse.ArgumentParser()
argparser.add_argument("--gpu", type=bool, default=True, help="Use GPU")
argparser.add_argument("--population", type=int, default=1000, help="Size of the population")
argparser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
argparser.add_argument("--dimension", type=int, default=40, help="Dimension of the problem")
argparser.add_argument("--kill_factor", type=int, default=0.6, help="How many parents kill")
argparser.add_argument("--min_iters", type=int, default=100, help="Minimal number of iterations to measure")
argparser.add_argument("--min_sec", type=int, default=10, help="Minimal running time of measurement")
argparser.add_argument("--max_sec", type=int, default=60*15, help="Maximum running time of measurement")
argparser.add_argument("--CR", type=float, default=0.4, help="Differential evolution CR term")
argparser.add_argument("--F", type=float, default=0.8, help="Differential evolution F term")
args, _ = argparser.parse_known_args()

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


populations = [
            10,         20,         30,         50,         75,
           100,        200,        300,        500,        750,
         1_000,      2_000,      3_000,      5_000,      7_500,
        10_000,     20_000,     30_000,     50_000,     75_000,
       100_000,    200_000,    300_000,    500_000,    750_000,
     1_000_000,  2_000_000,  3_000_000,  5_000_000,  7_500_000,
    10_000_000,
]
iterations = [
    100_000
]
functions = [
    create_fn1, #create_fn1mapfn
]
usegpu = [True, False]


@tf.function
def one_iteration(population, fn, pop):
    fitneses, real_fitness = fn(population)  # type: (tf.Tensor, tf.Tensor)
    num_parents = int(pop * args.kill_factor)
    parents_tournament_indices = tf.random.uniform(shape=(2, num_parents), minval=0, maxval=pop, dtype=tf.int32)
    comparison = tf.gather(fitneses, parents_tournament_indices[0], axis=0) < tf.gather(fitneses, parents_tournament_indices[1], axis=0)
    better = tf.concat([
        tf.boolean_mask(parents_tournament_indices[0], comparison, axis=0),
        tf.boolean_mask(parents_tournament_indices[1], tf.math.logical_not(comparison), axis=0)
    ], axis=0)
    parents = tf.gather(population, better, axis=0)

    # create children
    num_children = pop - num_parents
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

# warm up
print("Warm up")
for pop in [100, 200, 300, 400]:
    for gpu in usegpu:
        for create_fn in functions:
            fn = create_fn()
            with tf.device("/GPU:0" if gpu else "/CPU:0"):
                fn = create_fn()
                population = tf.random.uniform(shape=(pop, args.dimension), minval=-10, maxval=10)
                for iter in range(30):
                    population, progress = one_iteration(population, fn, pop)
print("Warmup done")


with open("benchmark_keras.txt", "w") as f:
    print("pop;iterations;function;gpu;s", file=f)
    for pop in populations:
        for i in iterations:
            for create_fn in functions:
                for gpu in usegpu:

                    fn = create_fn()
                    start = timer()
                    with tf.device("/GPU:0" if gpu else "/CPU:0"):
                        population = tf.random.uniform(shape=(pop, args.dimension), minval=-10, maxval=10)
                        relative_progress = np.zeros(shape=(i,))
                        print(f"Running fn {create_fn.__name__}, pop {pop}, iterations: {i} on {'GPU' if gpu else 'CPU'}")

                        for iter in progressbar(range(i)):
                            if iter > args.min_iters and timer() - start > args.min_sec or timer() - start > args.max_sec:
                                break
                            population, progress = one_iteration(population, fn, pop)
                            relative_progress[iter] = progress

                    end = timer()
                    running_time = end - start
                    executed_iters = iter
                    expected_iters = i
                    executed_iters_fraction = executed_iters / expected_iters
                    expected_running_time = running_time / executed_iters_fraction
                    print(f"Expected runnign time: {expected_running_time}s", flush=True)
                    print(f"{pop};{expected_iters};{create_fn.__name__};{1 if gpu else 0};{expected_running_time}", file=f, flush=True)

#plt.plot(relative_progress)
#plt.yscale('log')
#plt.show()
