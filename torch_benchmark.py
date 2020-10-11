import argparse
import torch
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
argparser.add_argument("--CR", type=float, default=0.4, help="Differential evolution CR term")
argparser.add_argument("--F", type=float, default=0.8, help="Differential evolution F term")
args, _ = argparser.parse_known_args()

dev_gpu = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
dev_cpu = torch.device("cpu")
print(f"Gonna use {dev_gpu}")

def create_fn1(gpu,popsize):
    xopt = torch.rand(size=(args.dimension,), dtype=torch.float, device=dev_gpu if gpu else dev_cpu) * 20.0 - 10.0
    fopt = np.random.rand() * 200.0 - 100.0
    dot_tmp = torch.zeros(size=(popsize, args.dimension), device=dev_gpu if gpu else dev_cpu)
    sum_tmp = torch.zeros(size=(popsize,), device=dev_gpu if gpu else dev_cpu)
    fitness_tmp = torch.zeros(size=(popsize,), device=dev_gpu if gpu else dev_cpu)
    def fn1(members: torch.Tensor):
        dot = torch.sub(members, xopt[None,:], out=dot_tmp)
        value = torch.sum(torch.mul(dot, dot, out=dot), dim=1, out=sum_tmp)
        return (torch.add(value, fopt, out=fitness_tmp), value)
    return fn1

def create_fn1numpy(gpu,popsize):
    xopt = np.random.random(size=(args.dimension,)) * 20.0 - 10.0
    fopt = np.random.rand() * 200.0 - 100.0
    dot_tmp = np.zeros(shape=(popsize, args.dimension))
    sum_tmp = np.zeros(shape=(popsize,))
    fitness_tmp = np.zeros(shape=(popsize,))
    torch_fitness_tmp = torch.zeros(size=(popsize,), device=dev_gpu if gpu else dev_cpu)
    torch_relative_tmp = torch.zeros(size=(popsize,), device=dev_gpu if gpu else dev_cpu)
    def fn1(members: torch.Tensor):
        dot = np.subtract(members.cpu().numpy(), xopt[None,:], out=dot_tmp)
        dot = np.multiply(dot, dot, out=dot)
        summed = np.sum(dot, axis=1, out=sum_tmp)
        fitness = np.add(summed, fopt, out=fitness_tmp)
        return (torch.from_numpy(fitness).to(torch_fitness_tmp), torch.from_numpy(summed).to(torch_relative_tmp))
    return fn1

def create_fn1unbind(gpu,popsize):
    xopt = torch.rand(size=(args.dimension,), dtype=torch.float, device=dev_gpu if gpu else dev_cpu) * 20.0 - 10.0
    fopt = np.random.rand() * 200.0 - 100.0
    torch_tmp = torch.zeros(size=(2, popsize), device=dev_gpu if gpu else dev_cpu)
    def axis_fn(example: torch.Tensor):
        val = (example - xopt) @ (example - xopt)
        return [val + fopt, val]
    def fn1(members: torch.Tensor):
        value = torch.Tensor([
            axis_fn(x_i) for i, x_i in enumerate(torch.unbind(members, dim=0), 0)
        ]).to(torch_tmp)
        return (torch_tmp[0], torch_tmp[1])
    return fn1

def create_fn1axisnumpy(gpu,popsize):
    xopt = np.random.random(size=(args.dimension,)) * 20.0 - 10.0
    fopt = np.random.rand() * 200.0 - 100.0
    torch_tmp = torch.zeros(size=(2, popsize), device=dev_gpu if gpu else dev_cpu)
    def axis_fn(example: np.ndarray):
        val = (example - xopt) @ (example - xopt)
        return (val + fopt, val)
    def fn1(members: torch.Tensor):
        examples = members.cpu().numpy()  # type: np.ndarray
        value = np.apply_along_axis(axis_fn, axis=1, arr=examples)
        value = torch.from_numpy(value).to(torch_tmp)
        return (torch_tmp[0], torch_tmp[1])
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
    100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000,
]
functions = [
    create_fn1, create_fn1numpy, create_fn1unbind, create_fn1axisnumpy
]
usegpu = [True, False]

with open("benchmark.txt", "w") as f:
    print("pop;iterations;function;gpu;s", file=f)
    for pop in populations:
        for i in iterations:
            for create_fn in functions:
                for gpu in usegpu:

                    fn = create_fn(gpu, pop)
                    start = timer()
                    dev = dev_gpu if gpu else dev_cpu
                    population = torch.rand(size=(pop, args.dimension), dtype=torch.float, device=dev)
                    relative_progress = np.zeros(shape=(i,))
                    print(f"Running fn {create_fn.__name__}, pop {pop}, iterations: {i} on {dev}")

                    num_parents = int(pop * args.kill_factor)
                    num_children = pop - num_parents
                    parent_tournament_indices_tmp = torch.zeros(size=(2, num_parents), dtype=torch.long, device=dev)
                    comparison_tmp = torch.zeros(size=(num_parents,), dtype=torch.bool, device=dev)
                    better_tmp = torch.zeros(size=(num_parents,), dtype=torch.long, device=dev)
                    picked_parents_tmp = torch.zeros(size=(4, num_children), dtype=torch.long, device=dev)
                    crossover_sample_tmp = torch.zeros(size=(num_children, args.dimension), device=dev)
                    crossover_sample_bool = torch.zeros(size=(num_children, args.dimension), dtype=torch.bool, device=dev)
                    mutated_tmp = torch.zeros(size=(num_children, args.dimension), device=dev)

                    for iter in progressbar(range(i)):

                        if iter > args.min_iters and timer() - start > args.min_sec:
                            break

                        fitneses, real_fitness = fn(population)  # type: (torch.Tensor, torch.Tensor)
                        # pickup parents
                        parents_tournament_indices = torch.randint(0, pop, size=(2, num_parents), out=parent_tournament_indices_tmp)
                        comparison = torch.le(fitneses[parents_tournament_indices[0]], fitneses[parents_tournament_indices[1]], out=comparison_tmp)
                        better = torch.cat([
                            parents_tournament_indices[0, comparison],
                            parents_tournament_indices[1, torch.logical_not(comparison, out=comparison)]
                        ], dim=0, out=better_tmp)
                        parents = population[better]

                        # create children
                        picked_parents = torch.randint(0, num_parents, size=[4, num_children], out=picked_parents_tmp)
                        crossover_sample = torch.gt(torch.rand(size=(num_children, args.dimension), device=dev, out=crossover_sample_tmp), args.CR, out=crossover_sample_bool)
                        mutated = torch.sub(parents[picked_parents[1]], parents[picked_parents[2]], out=mutated_tmp)
                        mutated = torch.mul(mutated, args.F, out=mutated)
                        mutated = torch.add(mutated, parents[picked_parents[0]], out=mutated)
                        mutated[crossover_sample] = parents[picked_parents[3]][crossover_sample]
                        mutated = torch.clamp(mutated, -10, 10, out=mutated)

                        population = torch.cat([parents, mutated], dim=0, out=population)
                        relative_progress[iter] = torch.mean(real_fitness, dim=0).item()

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
