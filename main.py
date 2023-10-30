import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


my_data = np.array([55, 67, 28, 235, 114])
x_axis = np.arange(len(my_data))
my_mean = my_data.mean()
my_stdev = my_data.std()


my_data_normed = (my_data - my_mean) / my_data.std()

my_data_domain_standardised = (my_data - my_data.min()) / (my_data.max() - my_data.min())


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5))
ax1.plot(x_axis,my_data)
ax1.set_title('Original')
ax2.plot(x_axis,my_data_normed)
ax2.set_title('Normalised \n(mean = 0, stddev = 1)')
ax3.plot(x_axis,my_data_domain_standardised)
ax3.set_title('Domain standardised \n(data points between 0 and 1)')
plt.show()


xs = [10, 100, 25, 64, 74]
ys = [125, 26, 66, 1, 10]

plt.scatter(xs, ys)

plt.show()


xys = [[10, 125], [100, 26], [25, 66], [67, 1], [74, 10]]
xys = np.array(xys)
plt.scatter(xys[:, 0], xys[:, 1])
plt.show()



#It's doing the mean of each column
mean = np.mean(xys, 0)
print(xys, mean)
#or this way
mn = xys.mean(0)

#It's doing the mean of each line
mean = np.mean(xys, 0)
print(xys, mean)

mean = np.mean( xys, 0)
std_dev = np.std(xys, 0)

ellipse = patches.Ellipse([mean[0], mean[1]], std_dev[0]*2, std_dev[1]*2, alpha=0.25)
fig, graph = plt.subplots()

graph.scatter(xys[:, 0], xys[:, 1])
graph.scatter(mean[0], mean[1])
graph.add_patch(ellipse)
plt.show()

dists = [np.linalg.norm(xy - mean) for xy in xys]
print(dists)

#Normalising the numbers in columns
x_min = np.min(xys,0)
x_max = np.max(xys, 0)
normed = (xys - x_min)/ (x_max - x_min)
print(normed)

mean_normed = np.mean(normed, 0)
std_dev_normed = np.std(normed, 0)

ellipse_normed = patches.Ellipse([mean_normed[0], mean_normed[1]], std_dev_normed[0]*2, std_dev_normed[1]*2, alpha=0.25)
fig, graph_normed = plt.subplots()

graph_normed.scatter(normed[:, 0], normed[:, 1])
graph_normed.scatter(mean_normed[0], mean_normed[1])
graph_normed.add_patch(ellipse_normed)
plt.show()

