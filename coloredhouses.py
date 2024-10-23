import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Define new colors for the plot
new_colors = ["xkcd:mocha", "Green", "xkcd:sky"]
new_colormaps = ["Reds", "Blues", "Greens"]
mymarkers = ['^', 'o', 'x']
level_markers = {'^': "$\lambda = 0$", 'o': '$\lambda = 1$', 'x': '$\lambda = 10$'}
algorithm_colors = {'xkcd:mocha': 'RPPerfGD', 'Green': 'RGD', 'xkcd:sky': 'RRGD'}

marker_handles = [mlines.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=10, label=label)
                  for marker, label in level_markers.items()]
color_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=label)
                 for color, label in algorithm_colors.items()]

def plot_traj(history, fig, ax, l, cma, co, subplot_idx, stddev):
    accuracy_history, theta_history = history
    accuracy_std, theta_std = stddev
    # Plot accuracy history with error bars
    ax.errorbar(range(len(accuracy_history)), accuracy_history, yerr=accuracy_std, ms=5, label=l, color=co, markevery=1, marker=cma)
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim([.947, .965])
    ax.legend()

epss = [0, 1, 10]

# Initialize figure with subplots
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

for k, eps in enumerate(epss):
    for l, (learner_name, color, marker) in enumerate(zip(['RPPerfGD', 'RGD', 'PiRGD'], new_colors, mymarkers)):
        avg_accuracies = np.load(f'accuracy_{learner_name}_eps_{eps}.npy')
        avg_theta_histories = np.load(f'theta_{learner_name}_eps_{eps}.npy')
        std_accuracies = np.load(f'std_accuracy_{learner_name}_eps_{eps}.npy')
        std_theta_histories = np.load(f'std_theta_{learner_name}_eps_{eps}.npy')
        
        # Plot trajectory for the current learner
        plot_traj((avg_accuracies, avg_theta_histories), fig, ax, f'{learner_name} eps={eps}', marker, color, k, (std_accuracies, std_theta_histories))

plt.tight_layout()
# Add legends with fontsize
first_legend = ax.legend(handles=marker_handles, title='shift', loc='upper left', fontsize=12, title_fontsize=14)
second_legend = ax.legend(handles=color_handles, title='Algorithms', loc='lower right', fontsize=12, title_fontsize=14)
ax.add_artist(first_legend)  # Add the first legend again so it appears along with the second one

plt.savefig('housesn_new_colors.pdf')
plt.show()
