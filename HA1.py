import numpy as np
import matplotlib.pyplot as plt

data_file_path = r"C:\Users\jose3\miniconda3\envs\MLA_prep\MNIST-5-6-Subset.txt"
data_matrix = np.loadtxt(data_file_path).reshape(1877, 784)
 # Load the labels from MNIST-5-6-Labels.txt
 # Change the path as needed
labels_file_path = r"C:\Users\jose3\miniconda3\envs\MLA_prep\MNIST-5-6-Subset-Labels.txt"
old_labels = np.loadtxt(labels_file_path)
#relabel our labels
labels = np.where(old_labels ==5, 1, -1)

def dist(training_points, test_points):
    #make the diagonals
    training_diag = np.diag(np.dot(training_points, training_points.T))
    test_diag = np.diag(np.dot(test_points, test_points.T)).T
    #multiply by 1's
    training_term = np.tile(training_diag, (test_diag.size, 1)).T
    test_term = np.tile(test_diag,(training_diag.size,1))
    #calc the cross term
    cross_term = 2*np.dot(training_points,test_points.T)
    return training_term+test_term-cross_term

def knn(training_points, test_points, training_labels, test_labels):
    #Get the distance matrix, accidently computed it transposed earlier <3
    distance_matrix = dist(training_points,test_points).T
    #Sort it, and remember the arguments
    distance_matrix_argsort = np.argsort(distance_matrix, axis=1)
    #Copy the labels and get matrices
    test_label_matrix = np.tile(test_labels,(training_labels.size,1)).T
    training_labels_matrix = np.tile(training_labels,(test_labels.size,1))
    #Sort the labels, acording to distance
    training_labels_sorted = np.take_along_axis(training_labels_matrix, distance_matrix_argsort, axis=1)
    test_labels_sorted = np.take_along_axis(test_label_matrix,distance_matrix_argsort, axis=1)
    #Use the cumulative sum, to see which labels fits the best.
    cumsum_matrix = np.cumsum(training_labels_sorted, axis=1)
    # if is value is below 0 it's -1, and above 0 it's 1 so we only need the signs.
    cumsum_sign = np.sign(cumsum_matrix)
    # the error is the sign of the absolute difference between cumsum_sign and test-labels sorted
    error=np.sign(np.abs(cumsum_sign-test_labels_sorted))
    #We then compute the mean error along the test_points
    mean_error=np.mean(error, axis=0)
    return mean_error

m = 50
sets = 5

training_points = data_matrix[:m,:]
training_labels = labels[:m]
validation_points = data_matrix[m:m+100,:]
validation_labels = labels[m:m+100]

test_points=data_matrix[m+100:m+200,:]
test_labels=labels[m+100:m+200]


print(knn(training_points,validation_points,training_labels,validation_labels))

indices = np.arange(m)

for j in range(4):
    n = 10*2**j
    error = []
    # make 5 validation sets
    for i in range(sets):
        point = data_matrix[m + i * n:m + (i + 1) * n, :]
        label = labels[m + i * n:m + (i + 1) * n]
        err = knn(training_points, point, training_labels, label)
        #plt.plot(indices,err,marker='o')
        error.append(err)

    # turn it into a matrix for variance computation
    error_matrix = np.array(error)

    # calculate variance in error
    variance = np.zeros(m)
    for i in range(m):
        var = np.var(error_matrix[:, i])
        variance[i] = var

    plt.plot(indices, variance, marker='o', label="n="+str(n) )

plt.xlabel('K')
plt.ylabel('variance over the 5 sets')
plt.title('var for each K, with different values for n')
plt.grid(True)
plt.legend(title="Size of validation sets")
plt.savefig("Variance over K, for different n")


for j in range(4):
    error = []
    n = 50*2**j
    for i in range(sets):
            point = data_matrix[m + i * n:m + (i + 1) * n, :]
            label = labels[m + i * n:m + (i + 1) * n]
            err = knn(training_points, point, training_labels, label)
            plt.plot(indices,err,marker='o')
            error.append(err)


    plt.xlabel('K')
    plt.ylabel('Avg error')
    plt.title('Avg error for each K with n ='+str(n))
    plt.grid(True)
    name = "Avg error for each K with n =" + str(n)
    plt.savefig(name)



