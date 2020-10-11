import argparse
import torch
import numpy as np
from progressbar import progressbar
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument("--gpu", type=bool, default=True, help="Use GPU")
argparser.add_argument("--population", type=int, default=1000, help="Size of the population")
argparser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
argparser.add_argument("--dimension", type=int, default=10, help="Dimension of the problem")
argparser.add_argument("--kill_factor", type=int, default=0.6, help="Dimension of the problem")
argparser.add_argument("--CR", type=float, default=0.4, help="Differential evolution CR term")
argparser.add_argument("--F", type=float, default=0.8, help="Differential evolution F term")
args, _ = argparser.parse_known_args()

dev = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
print(f"Gonna use {dev}")

def create_fn1():
    xopt = torch.rand(size=(args.dimension,), dtype=torch.float, device=dev) * 20.0 - 10.0
    fopt = np.random.rand() * 200.0 - 100.0
    def fn1(members: torch.Tensor):
        dot = members - xopt[None,:]
        value = torch.sum(dot * dot, dim=1)
        return (value + fopt, value)
    return fn1

def create_fn1numpy():
    xopt = np.random.random(size=(args.dimension,)) * 20.0 - 10.0
    fopt = np.random.rand() * 200.0 - 100.0
    def fn1(members: torch.Tensor):
        dot = members.cpu().numpy() - xopt[None,:]
        value = torch.from_numpy(np.sum(dot * dot, axis=1)).to(dev)
        return (value + fopt, value)
    return fn1

def create_fn1unbind():
    xopt = torch.rand(size=(args.dimension,), dtype=torch.float, device=dev) * 20.0 - 10.0
    fopt = np.random.rand() * 200.0 - 100.0
    def axis_fn(example: torch.Tensor):
        return (example - xopt) @ (example - xopt)
    def fn1(members: torch.Tensor):
        value = torch.stack([
            axis_fn(x_i) for i, x_i in enumerate(torch.unbind(members, dim=0), 0)
        ], dim=0).to(dev)
        return (value + fopt, value)
    return fn1

def create_fn1axisnumpy():
    xopt = np.random.random(size=(args.dimension,)) * 20.0 - 10.0
    fopt = np.random.rand() * 200.0 - 100.0
    def axis_fn(example: np.ndarray):
        return (example - xopt) @ (example - xopt) + fopt
    def fn1(members: torch.Tensor):
        examples = members.cpu().numpy()  # type: np.ndarray
        value = np.apply_along_axis(axis_fn, axis=1, arr=examples)
        value = torch.from_numpy(value).to(dev)
        return (value, value - fopt)
    return fn1

fn = create_fn1axisnumpy()


population = torch.rand(size=(args.population, args.dimension), dtype=torch.float, device=dev)
relative_progress = np.zeros(shape=(args.iterations,))
for iter in progressbar(range(args.iterations)):
    fitneses, real_fitness = fn(population)  # type: (torch.Tensor, torch.Tensor)
    # pickup parents
    num_parents = int(args.population * args.kill_factor)
    parents_tournament_indices = torch.randint(0, args.population, size=(2, num_parents), device=dev)
    comparison = fitneses[parents_tournament_indices[0]] < fitneses[parents_tournament_indices[1]]
    better = torch.cat([
        parents_tournament_indices[0, comparison],
        parents_tournament_indices[1, torch.logical_not(comparison)]
    ], dim=0)
    parents = population[better]

    # create children
    num_children = args.population - num_parents
    picked_parents = torch.randint(0, num_parents, [4, num_children])
    crossover_sample = torch.rand(size=(num_children, args.dimension), device=dev) > args.CR
    mutated = parents[picked_parents[0], :] + args.F * (parents[picked_parents[1]] - parents[picked_parents[2]])
    mutated[crossover_sample] = parents[picked_parents[3]][crossover_sample]
    mutated = torch.clamp(mutated, -10, 10, out=mutated)

    population = torch.cat([parents, mutated], dim=0, out=population)
    relative_progress[iter] = torch.mean(real_fitness, dim=0)


#plt.plot(relative_progress)
#plt.yscale('log')
#plt.show()
