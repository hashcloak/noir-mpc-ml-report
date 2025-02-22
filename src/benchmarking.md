## Benchmarking

We performed a benchmarking process to test the performance of our implementation. This benchmarking was done using two datasets: the [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) and the [Wine dataset](https://archive.ics.uci.edu/dataset/109/wine). In this benchmarking, we measured the following performance metrics:

- Accuracy with respect to a clear text implementation using floating-point numbers.
- Number of gates and ACIR opcodes of the Noir implementation for different number of samples.
- Training time using co-noir for different number of samples.

The benchmarks were executed in a server with an AMD EPYC Processor @ 2.0 GHz with 32 GB of RAM. 

### Datasets

We begin by describing the Iris dataset. The Iris dataset contains 50 samples for each type of Iris flower: *setosa*, *versicolor*, and *virginica*, having a total of 150 samples. For each sample, the dataset contains four features: the length and the width of the sepal and petal of each flower measured in centimeters. In this case, a logistic regression model will take the length and the with of the sepal and petal for a new length in centimeters, and the model will tell whether this flower is a *setosa*, a *versicolor*, or a *virginica* type.

On the other hand, the Wine dataset contains a total of 178 samples, where each sample is one of three types of wines grown in the same region of Italy but derived from three different cultivars. Each sample in the dataset has 13 features presented next:

1) Alcohol
2) Malic acid
3) Ash
4) Alcalinity of ash
5) Magnesium
6) Total phenols
7) Flavanoids
8) Nonflavanoid phenols
9) Proanthocyanins
10) Color intensity
11) Hue
12) OD280/OD315 of diluted wines
13) Proline 

### Results

For the Iris dataset, we obtained the following results for the number of gates and ACIR opcodes:

| Epochs | Train samples | ACIR opcodes | Circuit size | Proving time |
|:------:|:-------------:|:------------:|:------------:|--------------|
| 10     | 30            | 317,088      | 660,166      | 0m 39.295s   |
| 10     | 50            | 523,048      | 1,085,961    | 1m 15.344s   |
| 20     | 30            | 655,848      | 1,355,005    | 1m 18.012s   |
| 20     | 50            | 1,082,008    | 2,232,450    | 2m 35.643s   |
| 30     | 30            | 994,608      | 2,049,841    | 1m 24.117s   |
| 30     | 50            | 1,640,968    | 3,378,936    | 2m 19.931s   |

For the Wine dataset, we obtained the following results for the number of gates and ACIR opcodes:

| Epochs | Train samples | ACIR opcodes | Circuit size | Proving time |
|:------:|:-------------:|:------------:|:------------:|--------------|
| 10     | 30            | 614,088      | 1,402,019    | 1m 19.345s   |
| 10     | 50            | 1,007,788    | 2,301,844    | 2m 37.078s   |
| 20     | 30            | 1,260,378    | 2,866,087    | 2m 18.731s   |
| 20     | 50            | 2,068,678    | 4,708,962    | 1m 49.081s   |
| 30     | 30            | 1,906,668    | 4,330,154    | 1m 32.140s   |

For the last table with the Wine dataset, we did not measure the case for 30 epochs and 50 training samples given that it fills the RAM memory.

In the case of the co-noir training time, we have the following results for the Iris dataset:

| Epochs | Train samples | Training time [sec.] |
|--------|---------------|----------------------|
|     10 |            30 |                2,040 |
|     10 |            50 |                3,545 |
|     20 |            30 |                4,148 |

The case for 20 epochs and 50 samples was not possible to run because the generation of the witness takes too long and the co-noir process gets killed because of time out.

For the Wine dataset with 10 epochs and 30 training samples, it takes 4,365 seconds.

As a reference, a Python training using `scikit-learn` for 30 and 50 samples takes around ~0.006 seconds in average (yes, there is not much difference between them) using a laptop with 20 × 13th Gen Intel® Core™ i7-13700H with 32 GB of RAM. Although it is well understood that it is not possible (or at least very, ***VERY*** difficult) to obtain a training time similar to a clear text implementation, this shows that there is a lot of work to do in the realm of privacy-preserving machine learning to improve the performance of this kind of protocols. However, when protocols are running in servers, it is possible to increase the capabilities of the servers to speed-up the training and proving process using co-noir without sacrificing or compromising the security guarantees.

Finally, we compared our Noir implementation using fixed-point numbers with a Rust implementation using floating-point numbers with type `f64`. We found that both implementations obtain ***exactly*** the same accuracy in all the examples we ran. This means that the fact that we are using fixed point numbers in the secure training does not affect significantly the result with respect to a floating point training. To reproduce this experiments, you can use the logistic regression implementation in Rust presented in [this repository](https://github.com/ewynx/rs-logistic-regression). You can use the Rust implementation along with the scripts `run_single_test.sh`, `accuracy_evaluation/evaluate_float_model.py` and `accuracy_evaluation/generate_rust_dataset.py` to compare both accuracies.

These numbers can be found in the [Benchmark report](https://docs.google.com/spreadsheets/d/1H9VOBIaQpKs2oqNCLjtJT7AEs75i7uHKTq4Re4_cD-o/edit?usp=sharing) and reproduced using the [noir-mpc-ml benchmarking repo](https://github.com/hashcloak/bench-noir-mpc-ml).
