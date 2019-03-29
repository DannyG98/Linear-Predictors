import numpy as np
np.set_printoptions(suppress=True)

def load_data(num_rows=-1):
    if num_rows >= 0:
        return np.genfromtxt('Breast_cancer_data.csv', delimiter=',', skip_header=1, max_rows=num_rows)
    else:
        return np.genfromtxt('Breast_cancer_data.csv', delimiter=',', skip_header=1)


def pre_processing(data):
    malignant = []
    benign = []

    for row in data:
        if row[-1] == 0:
            benign.append(row[:-1])
        else:
            malignant.append(row[:-1])

    malignant = np.array(malignant)
    benign = np.array(benign)
    return benign, malignant


def eigen_decomp(matrix):
    eigenvalues, eigenvector = np.linalg.eig(matrix)
    eigenvector_T = eigenvector.transpose()

    return eigenvector, eigenvalues, eigenvector_T


def create_bmatrix(data):
    x_matrix = []
    y_matrix = []

    for row in data:
        x_matrix.append(row[:-1])
        y_matrix.append([row[-1]] if row[-1] != 0 else [-1])

    x_matrix = np.matrix(x_matrix).transpose()
    y_matrix = np.matrix(y_matrix)

    return x_matrix * y_matrix


def create_amatrix(data):
    a_matrix = []

    for row in data:
        a_matrix.append(row[:-1])

    a_matrix = np.matrix(a_matrix)
    a_matrix_T = a_matrix.transpose()

    return a_matrix_T * a_matrix


def create_Dplus(eigenvector):
    Dplus_matrix = []
    for x in eigenvector:
        Dplus_value = 1 / x if x != 0 else 0
        Dplus_matrix.append(Dplus_value)

    Dplus_matrix = np.matrix(np.diag(Dplus_matrix))

    return Dplus_matrix


def linear_regression(data):
    A_matrix = create_amatrix(data)
    B_matrix = create_bmatrix(data)

    eigenvector, eigenvalue, eigenvector_T = eigen_decomp(A_matrix)
    eigenvalue_plus = create_Dplus(eigenvalue)

    A_plus = eigenvector * eigenvalue_plus * eigenvector_T

    return A_plus * B_matrix


def calculate_distance(hyperplane, point, bias=0):
    numerator = abs(np.dot(hyperplane, point) + bias)
    denominator = np.linalg.norm(hyperplane)

    return numerator/denominator


def find_error(data, zero_plane, one_plane):
    num_errors = 0

    zero_plane = np.asarray(zero_plane.transpose())[0]
    zero_plane_bias = zero_plane[0]
    zero_plane = zero_plane[1:]

    one_plane = np.asarray(one_plane.transpose())[0]
    one_plane_bias = one_plane[0]
    one_plane = one_plane[1:]

    for row in data:
        modified_row = row[:-1]
        target = row[-1]

        zero_dist = calculate_distance(zero_plane, modified_row[:-1], bias=zero_plane_bias)
        one_dist = calculate_distance(one_plane, modified_row[:-1], bias=one_plane_bias)

        prediction = 0.0 if zero_dist < one_dist else 1.0

        if prediction != target:
            num_errors += 1

    return num_errors / len(data)


if __name__ == '__main__':
    df = load_data()
    zero_df, one_df = pre_processing(df)

    zero_bias = np.ones(shape=(len(zero_df), 1))
    one_bias = np.ones(shape=(len(one_df), 1))

    zero_df = np.append(zero_bias, zero_df, axis=1)
    one_df = np.append(one_bias, one_df, axis=1)

    zero_line = linear_regression(zero_df)
    one_line = linear_regression(one_df)

    print("Weight Vector is formatted as follows: [bias, mean_radius, mean_texture, mean_perimeter, mean_area]")
    print("Weight Vectors predict on the mean_smoothness feature")
    print("Weight Vector for Benign")
    print(zero_line)

    print("Weight Vector for Malignant")
    print(one_line)

    print("Loss is:", find_error(df, zero_line, one_line))
