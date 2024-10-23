import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from learner import *
from dataset import *
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

cos = ["xkcd:wintergreen", "Blue", "Green", "Yellow", "Purple", "Orange", "Pink", "Gray", "Brown"]
cmas = ["Reds", "Blues", "Greens", "YlOrBr", "Purples", "Oranges", "Pink", "cool", "Greys", "YlOrBr"]

mymarkers=['^', 'o', 'x']
level_markers = {'^': "$\lambda = 0$", 'o': '$\lambda = 1$', 'x': '$\lambda = 10$'}
algorithm_colors = {'xkcd:wintergreen': 'RRM'}

marker_handles = [mlines.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=10, label=label)
                  for marker, label in level_markers.items()]
color_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=label)
                 for color, label in algorithm_colors.items()]
def plot_traj(history, fig, ax, l, cma, co, subplot_idx, stddev):
    accuracy_history, theta_history = history
    accuracy_std, theta_std = stddev
    ax_acc, ax_dist = ax[subplot_idx]

    # Plot accuracy history with error bars
    ax_acc.errorbar(range(len(accuracy_history)), accuracy_history, yerr=accuracy_std, fmt='-o', ms=2, label=l, color=co, markevery=10)
    ax_acc.set_title('Accuracy Across Iterations')
    ax_acc.set_xlabel('Iteration')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()

    distances = np.linalg.norm(theta_history - theta_history[-1], axis=1)
    distance_std = np.std([np.linalg.norm(t - theta_history[-1], axis=1) for t in theta_histories], axis=0)
    # Plot the distances with error bars
    ax_dist.errorbar(range(len(distances)), distances, yerr=distance_std, ms=2, linestyle='-', alpha=0.7, color=co, label=l, marker=cma)
    ax_dist.set_title('Distance to Final Theta Across Iterations')
    ax_dist.set_xlabel('Iteration')
    ax_dist.set_ylabel('Distance')

def plot_traj(history, fig, ax, l, cma, co, subplot_idx, stddev):
    accuracy_history, theta_history = history
    accuracy_std, theta_std = stddev
    # Plot accuracy history with error bars
    ax.errorbar(range(len(accuracy_history)), accuracy_history, yerr=accuracy_std, ms=5, label=l, color=co, markevery=1, marker=cma)
    #ax.set_title('Accuracy Across Iterations')
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    #ax.set_ylim([.947, .965])
    ax.legend()


num_iter = 15
n = 18000
seed = 0
#epss=[0, 1, 2]
n_runs = 10
epss = [0, 1, 10]
#epss = [0, 0.01, 0.1]

# Initialize figure with subplots
#fig, ax = plt.subplots(len(epss), 2, figsize=(6, 4 * len(epss)))
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# To store all accuracy results
all_accuracies = []

for k, eps in enumerate(epss):
    u = np.zeros(8)
    u[[0,4,5,6]] = eps

    Pi = np.diag(u)
    X_init, y_init = perf_houses(n=n, seed=seed)

    eps_accuracies = []
    eps_theta_histories = []

    for l,learner_class in enumerate([RRM]):
        accuracies = []
        theta_histories = []

        for run in range(n_runs):
            learner = learner_class(Pi=Pi, step_size=.2)
            learner.init_model(X_init, y_init)
            for i in tqdm(range(num_iter)):
                X, y = perf_houses(n=n, theta=learner.get_theta(), seed=seed + i + run * 100+l*2000, eps=eps)
                learner.grad(X, y)
                learner.evaluate(X, y)
            acc_hist, theta_hist = learner.history()
            accuracies.append(acc_hist)
            theta_histories.append(theta_hist)

        # Average the results over 10 runs
        avg_accuracies = np.mean(accuracies, axis=0)
        avg_theta_histories = np.mean(theta_histories, axis=0)
        std_accuracies = np.std(accuracies, axis=0)
        std_theta_histories = np.std(theta_histories, axis=0)
        eps_accuracies.append(avg_accuracies)
        eps_theta_histories.append(avg_theta_histories)

        learner_name = learner_class.__name__
        # Save the accuracy and theta history for this learner and epsilon
        np.save(f'accuracy_{learner_name}_eps_{eps}.npy', avg_accuracies)
        np.save(f'theta_{learner_name}_eps_{eps}.npy', avg_theta_histories)
        np.save(f'std_accuracy_{learner_name}_eps_{eps}.npy', std_accuracies)
        np.save(f'std_theta_{learner_name}_eps_{eps}.npy', std_theta_histories)

        # Plot trajectory for the current learner
        plot_traj((avg_accuracies, avg_theta_histories), fig, ax, f'{learner_name} eps={eps}', mymarkers[k], cos[l], k, (std_accuracies, std_theta_histories))

    all_accuracies.append(eps_accuracies)
plt.tight_layout()
# Add legends with fontsize
first_legend = ax.legend(handles=marker_handles, title='shift', loc='upper left', fontsize=12, title_fontsize=14)
second_legend = ax.legend(handles=color_handles, title='Algorithm', loc='lower right', fontsize=12, title_fontsize=14)
ax.add_artist(first_legend)  # Add the first legend again so it appears along with the second one

plt.savefig('housesrrm.pdf')
plt.show()