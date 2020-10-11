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
argparser.add_argument("--kill_factor", type=int, default=0.6, help="Dimension of the problem")
argparser.add_argument("--CR", type=float, default=0.4, help="Differential evolution CR term")
argparser.add_argument("--F", type=float, default=0.8, help="Differential evolution F term")
args, _ = argparser.parse_known_args()

dev_gpu = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
dev_cpu = torch.device("cpu")
print(f"Gonna use {dev_gpu}")

def create_fn1(gpu):
    xopt = torch.rand(size=(args.dimension,), dtype=torch.float, device=dev_gpu if gpu else dev_cpu) * 20.0 - 10.0
    fopt = np.random.rand() * 200.0 - 100.0
    def fn1(members: torch.Tensor):
        dot = members - xopt[None,:]
        value = torch.sum(dot * dot, dim=1)
        return (value + fopt, value)
    return fn1

def create_fn1numpy(gpu):
    xopt = np.random.random(size=(args.dimension,)) * 20.0 - 10.0
    fopt = np.random.rand() * 200.0 - 100.0
    def fn1(members: torch.Tensor):
        dot = members.cpu().numpy() - xopt[None,:]
        value = torch.from_numpy(np.sum(dot * dot, axis=1)).to(dev_gpu if gpu else dev_cpu)
        return (value + fopt, value)
    return fn1

def create_fn1unbind(gpu):
    xopt = torch.rand(size=(args.dimension,), dtype=torch.float, device=dev_gpu if gpu else dev_cpu) * 20.0 - 10.0
    fopt = np.random.rand() * 200.0 - 100.0
    def axis_fn(example: torch.Tensor):
        return (example - xopt) @ (example - xopt)
    def fn1(members: torch.Tensor):
        value = torch.stack([
            axis_fn(x_i) for i, x_i in enumerate(torch.unbind(members, dim=0), 0)
        ], dim=0).to(dev_gpu if gpu else dev_cpu)
        return (value + fopt, value)
    return fn1

def create_fn1axisnumpy(gpu):
    xopt = np.random.random(size=(args.dimension,)) * 20.0 - 10.0
    fopt = np.random.rand() * 200.0 - 100.0
    def axis_fn(example: np.ndarray):
        return (example - xopt) @ (example - xopt) + fopt
    def fn1(members: torch.Tensor):
        examples = members.cpu().numpy()  # type: np.ndarray
        value = np.apply_along_axis(axis_fn, axis=1, arr=examples)
        value = torch.from_numpy(value).to(dev_gpu if gpu else dev_cpu)
        return (value, value - fopt)
    return fn1


populations = [
            10,         20,         30,         50,         75,
           100,        200,        300,        500,        750,
         1_000,      2_000,      3_000,      5_000,      7_500,
        10_000,     20_000,     30_000,     50_000,     75_000,
       100_000,    200_000,    300_000,    500_000,    750_000,
     #1_000_000,  2_000_000,  3_000_000,  5_000_000,  7_500_000,
    #10_000_000,
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
    for pop in reversed(populations):
        for i in iterations:
            for create_fn in functions:
                for gpu in usegpu:

                    fn = create_fn(gpu)
                    start = timer()
                    dev = dev_gpu if gpu else dev_cpu
                    population = torch.rand(size=(pop, args.dimension), dtype=torch.float, device=dev)
                    relative_progress = np.zeros(shape=(i,))
                    for iter in progressbar(range(i)):
                        fitneses, real_fitness = fn(population)  # type: (torch.Tensor, torch.Tensor)
                        # pickup parents
                        num_parents = int(pop * args.kill_factor)
                        parents_tournament_indices = torch.randint(0, pop, size=(2, num_parents), device=dev)
                        comparison = fitneses[parents_tournament_indices[0]] < fitneses[parents_tournament_indices[1]]
                        better = torch.cat([
                            parents_tournament_indices[0, comparison],
                            parents_tournament_indices[1, torch.logical_not(comparison)]
                        ], dim=0)
                        parents = population[better]

                        # create children
                        num_children = pop - num_parents
                        picked_parents = torch.randint(0, num_parents, [4, num_children])
                        crossover_sample = torch.rand(size=(num_children, args.dimension), device=dev) > args.CR
                        mutated = parents[picked_parents[0], :] + args.F * (parents[picked_parents[1]] - parents[picked_parents[2]])
                        mutated[crossover_sample] = parents[picked_parents[3]][crossover_sample]
                        mutated = torch.clamp(mutated, -10, 10, out=mutated)

                        population = torch.cat([parents, mutated], dim=0, out=population)
                        relative_progress[iter] = torch.mean(real_fitness, dim=0)
                    end = timer()
                    print(f"{pop};{i};{create_fn.__name__};{1 if gpu else 0};{end-start}", file=f, flush=True)

#plt.plot(relative_progress)
#plt.yscale('log')
#plt.show()
