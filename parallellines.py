import matplotlib.pyplot as plt
import numpy as np

x = np.random.uniform(0, 1, 20)
y = np.random.uniform(0, 1, 20)

fig, ax = plt.subplots()

ax.scatter(np.zeros(20), x)
ax.scatter([0.3] * 20, y)

ax.set_xlim([-0.1, 0.4])
ax.set_xticks([0, 0.3])
ax.set_yticks([0, 1])

ax.set_xticklabels(['0', r'$\alpha$'])

plt.savefig("parallel.pdf")
plt.show()
