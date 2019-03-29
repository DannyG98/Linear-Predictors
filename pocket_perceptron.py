import numpy as np
np.set_printoptions(suppress=True)


def load_data(num_rows=-1):
    if num_rows >= 0:
        return np.genfromtxt('Breast_cancer_data.csv', delimiter=',', skip_header=1, max_rows=num_rows)
    else:
        return np.genfromtxt('Breast_cancer_data.csv', delimiter=',', skip_header=1)


def classify(modified_data, weights):
    return 1 if np.dot(modified_data, weights) > 0 else 0


def pre_processing(data):
    processed_data = []

    for row in data:
        processed_row = np.append([1], row)
        processed_data.append(processed_row)

    return np.array(processed_data)


def perceptron(data, max_iter=-1, bias=0, growth_rate=1.0):
    weights = np.array([0.0] * len(data[0]))
    iter_count = 0

    # The "Pocket" Aspect
    pocket_weight = weights
    pocket_loss = len(data)

    data = pre_processing(data)

    while iter_count != max_iter and pocket_loss != 0:
        for row in data:
            target = row[-1]
            feature_row = row[:-1]
            prediction = classify(feature_row, weights)

            # Check for error
            if target != prediction:
                # Convert from {0,0} to {-1,1}
                target = -1 if target == 0 else 1
                xy = target * feature_row
                xy *= growth_rate
                new_weight = np.add(weights, xy)
                weights = new_weight

                # The "Pocket" aspect
                current_loss = find_errors(weights, data)

                if pocket_loss > current_loss:
                    pocket_weight = weights
                    pocket_loss = current_loss

        iter_count += 1

        # THIS PORTION OF CODE BELONGED TO ORIGINAL PERCEPTRON ALGORITHM
        # # Check if there has been no updates to weight vector
        # if num_errors == 0:
        #     percent_error = find_loss(weights, data)
        #     return pocket_weight
        #
        # # This portion executes only if loss_bound is set
        # if loss_bound != 0.0:
        #     percent_error = find_loss(weights, data)
        #
        #     # THIS LINE IS FOR DEBUGGING/PERFORMANCE ANALYSIS PURPOSES ONLY
        #     # print("Iteration:", iter_count, "   Error:", percent_error)
        #
        #     # Terminates the learning process if loss is <= specified error bound
        #     if percent_error <= loss_bound:
        #         return pocket_weight

    return pocket_weight


def find_errors(weights, data):
    num_errors = 0

    for row in data:
        target = row[-1]
        feature_row = row[:-1]
        prediction = classify(feature_row, weights)

        if target != prediction:
            num_errors += 1

    return num_errors


def test(weights, data, print_error=False):
    num_errors = 0
    data = pre_processing(data)

    for row in data:
        target = row[-1]
        feature_row = row[:-1]
        prediction = classify(feature_row, weights)

        if target != prediction:
            if print_error:
                print("Data is:", row)
                print("Predicted:", prediction)
                print("Actual:", target)
            num_errors += 1

    print("Total Errors:", num_errors)
    return num_errors/len(data)


if __name__ == '__main__':
    df = load_data()
    print(len(df), 'rows')
    perceptron_model = perceptron(df, max_iter=100)
    print(perceptron_model)
    print('Percent Loss:', test(perceptron_model, df))
