import matplotlib.pyplot as plt


def make_distribution(data, args, phase):
    plt.figure(figsize=(25, 5))
    plt.bar(range(len(data)), [k for v, k in data.items()])
    plt.xticks(range(len(data)), [v for v, k in data.items()], rotation=60, fontsize=10)
    plt.title(f"data distribution of {args.data_type}_{args.use_index}_index in phase {phase}")
    plt.show()


def draw_results(data, args, category):
    plt.figure(figsize=(25, 5))
    plt.bar(range(len(data)), data)
    plt.xticks(range(len(data)), category, rotation=60, fontsize=10)
    plt.title(f"results distribution of {args.data_type}_{args.use_index}_index")
    plt.show()