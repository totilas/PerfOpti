from learner import *
from dataset import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
num_iter = 25
n = 1000
scale_0 = .5
seed = 0

levels = [.3, .55, 1]
all_groups = [None, None, None]

Pi = np.diag([.1, 3, 0, 0, 0, 0, 0])
mu = np.array([1, 2, 0.5, 0.5, 0, 0, 0]) 

theta = np.zeros(7)

cmap = plt.cm.get_cmap('plasma')
cmap2 = plt.cm.get_cmap('winter')

num_iter_average = 100

seeds = []
Xs = []

for j, level in enumerate(levels):

    learner_groups = [
        [RPPerfGD2(Pi=Pi, step_size=.1, Pi_learning=False) for _ in range(num_iter_average)],
        [SFPerfGD2(Pi=Pi, step_size=.1, Pi_learning=False, sigma=scale_0 * level) for _ in range(num_iter_average)],
        [PiRGD2(Pi=Pi, step_size=.1, Pi_learning=False) for _ in range(num_iter_average)],
        [PiRGD2(Pi=Pi, step_size=.1, Pi_learning=False, lamb=1e-1) for _ in range(num_iter_average)]
    ]

    # Initialize each learner
    for group in learner_groups:
        for k, learner in enumerate(group):
            X_init, y_init = generate_synthetic_data(n, scale=scale_0 * level, Pi=Pi, mu=mu, rng=np.random.RandomState(seed + k * num_iter), theta=theta)
            learner.init_model(X_init, y_init)

    for group in learner_groups:
        for k, learner in enumerate(group):
            learner_name = type(learner).__name__ + str(level)
            pca = PCA(n_components=2)
            pca.fit(X_init, y_init)
            X_init_pca = pca.transform(X_init)

            for i in (pbar := tqdm(range(num_iter))):
                pbar.set_description(learner_name)
                seeds.append(seed + i + k * num_iter)
                X, y = generate_synthetic_data(n, learner.get_theta(), scale=scale_0 * level, rng=np.random.RandomState(seed + i + 1 + k * num_iter), Pi=Pi, mu=mu)
                Xs.append(X)
                learner.grad(X, y)
                learner.evaluate(X, y)

            # X_pca = pca.transform(X)
            # plt.scatter(X_init_pca[:, 0], X_init_pca[:, 1], c=y_init, marker='o', label='Initial Dataset', cmap=cmap2)
            # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, marker='x', label='Last Dataset', cmap=cmap2)
            # plt.title(learner_name)
            # plt.xlabel("PC 1")
            # plt.ylabel("PC 2")
            # plt.legend()
            # plt.show()

    all_groups[j] = learner_groups

print(np.mean(np.std(np.stack(Xs, axis=2), axis=2), axis=0))
print(len(set(seeds)))

def plot_traj(histories, fig, ax, l, cma, co):
    accuracy_history = []
    theta_history = []
    for history in histories:
        acc, theta = history
        accuracy_history.append(np.array(acc))
        theta_history.append(np.array(theta))
    accuracy_history = np.stack(accuracy_history, axis=1)
    
    accuracy_history_mean = np.mean(accuracy_history, axis=1)
    accuracy_history_std = np.std(accuracy_history, axis=1)

    theta_history = np.stack(theta_history, axis=2)
    theta_history_mean = np.mean(theta_history, axis=2)
    theta_history_std = np.std(theta_history, axis=2)
    ax.errorbar(
        np.arange(len(accuracy_history_mean)),
        accuracy_history_mean,
        accuracy_history_std,
        marker=cma,
        linestyle='-',
        ms=5,
        label=l,
        color=co,
        markevery=1,
        errorevery=3
    )
    #ax.set_title('Accuracy Across Iterations')
    ax.set_ylim([.6, 1])
    ax.set_xlabel('Iterations', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.legend()


# Custom legend handles
level_markers = {'^': "$\sigma = .15$", 'o': '$\sigma = .275$', 'x': '$\sigma = .5$'}
algorithm_colors = {'xkcd:mocha': 'RPPerfGD', 'Green': 'RGD', 'xkcd:sky': 'RRGD', 'xkcd:purpley pink': 'SFPerfGD'}


cos = ["xkcd:mocha", 'xkcd:purpley pink', "Green", 'xkcd:sky', "Red", "Yellow", "Purple", "Orange", "Pink", "Cyan", "Gray", "Brown"]


cmas = ["^", "o", "x", "YlOrBr", "Purples", "Oranges", "seismic", "cool", "Greys", "YlOrBr"]

marker_handles = [mlines.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=5, label=label)
                  for marker, label in level_markers.items()]
color_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=5, label=label)
                 for color, label in algorithm_colors.items()]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
for j, level in enumerate(levels):
    for i, group in enumerate(all_groups[j]):
        learner_name = type(group[0]).__name__ + str(level)
        plot_traj(
            [learner.history() for learner in group],
            fig,
            ax,
            learner_name,
            cmas[j],
            cos[i]
        )
plt.grid()
plt.tight_layout()


# Add legends
first_legend = plt.legend(handles=marker_handles, loc='lower left', fontsize=14)
ax.add_artist(first_legend)
plt.legend(handles=color_handles,  loc='lower right', fontsize=14)

plt.savefig("sfvsrp.pdf")
plt.show()
