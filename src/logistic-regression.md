## Logistic regression

Now that we have the most important building block down, we can move on to the next challenge: implementing logistic regression in Noir.

Logistic regression is a classification algorithm in ML that estimates the probability of a sample belonging to a specific class, such as Class A or Class B. For example, in the case of medical images, logistic regression can determine whether cancer is present or not, with the classes being "Yes" and "No". 

There are two important phases in machine learning: the training and inference phase. In the first one, you use a dataset that links samples to the output classes to calculate the parameters of a prediction model. Samples will have a certain amount of features, which make them unique, and they have a label that indicates to which class they belong. For example, for a medical image, we don't use the image directly, but certain numerical representations of aspects of the image. The labels will be "cancer" and "no cancer". The weights represent the output of your training (you "trained a model"). Then, in the second phase, the inference, you take a sample that was not in the dataset and by doing a calculation which involves the weights, you predict in which class it falls. 

In this project, we have only focused on the training phase and specifically by implementing logistic regression. In this algorithm, we use the following strategy: make a guess using the sigmoid function, determine how "bad" or "good" that guess was using a loss function, and iteratively updates the weights to keep improving the model. 

Now, let's introduce the actual logistic regression model. Let $\mathbf{x}$ be a sample represented as a vector of features in the dataset of breast cancer, let's encode "Cancer" as 1 and "Not cancer" as 0, and let's call this response variable as $y$. We want to estimate $p(\mathbf{x}) = P(y = 1 \vert \mathbf{x})$. The estimation of this probability can be done using the following model:

$$
p(\mathbf{x}; \mathbf{w}, w_0) = \sigma(\mathbf{w}^\top \mathbf{x} + w_0),
$$

where $\mathbf{w}, w_0$ are the parameters of the model that we want to estimate, and $\sigma$ is the sigmoid function given by

$$
\sigma(x) = \frac{e^x}{1 + e^x}.
$$

The goal in the training process is to estimate the parameters $\mathbf{w}, w_0$ given a training dataset. This estimation can be done using the gradient descent algorithm. The algorithms to find the parameters is presented next: let $E$ be the number of epochs, $n$ be the number of samples, $m$ be the number of features, and $\alpha$ the learning rate.

**Algorithm for logistic regression:**

1. Initialize $\mathbf{w}$ and $w_0$ with zeroes.
2. Repeat $E$ times:
        1. For each $j$ in $m$: $w_j \leftarrow w_j - \alpha \cdot \frac{1}{n} \sum_{i=1}^n [p(\mathbf{x}^{(i)}; \mathbf{w}, w_0) - y] \cdot \mathbf{x}^{(i)}_j$
        2. $w_0 \leftarrow w_0 - \alpha \cdot \frac{1}{n} \sum_{i = 1}^n [p(\mathbf{x}^{(i)}; \mathbf{w}, w_0) - y]$.
3. Output $\mathbf{w}$ and $w_0$.

The trainin algorithm is implemented [here](https://github.com/hashcloak/noir-mpc-ml/blob/b008e225c6dcceabbcc0ddcf45e0a3fb6aaca935/src/ml.nr#L55-L91) in the repository.

### Sigmoid function

The sigmoid function that we mentioned before is defined as $\sigma(x) = \frac{1}{1 + e^{-x}}$. This function is an important subfunction of logistic regression and well, it looks pretty bad to implement in a zk language as Noir. 

But not to worry! From the ML functionality in MPC library [MP-SPDZ](https://github.com/data61/MP-SPDZ) we realized that this could be done differently, namely by approximation. Implemented [here](https://github.com/data61/MP-SPDZ/blob/master/Compiler/ml.py#L110) in MP-SPDZ, this approach was published by Hong et al. in [this paper](https://arxiv.org/abs/2002.04344) and does the following:

$$
\sigma(x) =
\begin{cases} 
10^{-4}, & x \leq -5 \\
0.02776 \cdot x + 0.145, & -5 < x \leq -2.5 \\
0.17 \cdot x + 0.5, & -2.5 < x \leq 2.5 \\
0.02776 \cdot x + 0.85498, & 2.5 < x \leq 5 \\
1 - 10^{-4}, & x > 5
\end{cases}
$$

In our library this is implemented [here](https://github.com/hashcloak/noir-mpc-ml/blob/master/src/ml.nr#L4). 

### Input sample bitsize restrictions

As discussed in the previous section, the underlying `Quantized` type does not come without its restrictions. In particular, we are dealing with limits on the amount of bits the (intermediate) values in the operations can occupy. To make sure none of the intermediate values overflow, there are assertions on bitsize of values throughout the various functions. All of these can be found in the `ml.nr` file of the library, but here we want to point out an important limit on the input samples.

The function `train_multi_class` is the main entry point to the library and it takes the following arguments:
```rust=
epochs: u64,
inputs: [[Quantized; M]; N],
labels: [[Quantized; N]; C],
learning_rate_ratio: Quantized
```
The following restrictions are important:
- inputs: max 20 bits
- labels: max 17 bits (they should be either 0 or 1, and represented in `Quantized`, this takes max 17 bits)
- learning_rate_ratio = learning_rate * ratio: max 11 bits. This aims to support values up to $0.1*0.1=0.01$ in decimal numbers

The bitsize limit imposed on the input samples is a choice that aligns with the datasets we're testing with for this project, as well as a tradeoff between allowing for more precision versus more iterations. If the dataset you are working with is not capped by 20 bits in quantized representation, it is possible to normalize all input samples (so they have values between 0 and 1, or -1 and 1), which will make them no more than 17 bits long. 

### Train your own model

To train a model with your data, you first need to import the library in the `Nargo.toml` file as follows:

```toml
[dependencies]
noir_mpc_ml = { git = "https://github.com/hashcloak/noir-mpc-ml", tag = "v0.1.2" }
```

Suppose that you want to train a logistic model with a dataset with 30 samples, 4 features and 3 possible classes for classification. Then, your code should look like this:

```rust
use noir_mpc_ml::ml::train_multi_class;
use noir_mpc_ml::quantized::Quantized;

fn main(
    epochs: u64,
    data: [[Quantized; 4]; 30],
    labels: [[Quantized; 30]; 3],
    learning_rate_ratio: Quantized,
    parameters: pub [([Quantized; 4], Quantized); 3],
) {
    let parameters_train = train_multi_class(epochs, data, labels, learning_rate_ratio);
    assert(parameters == parameters_train);
}
```

Let us explain the parameters for the `main()` function:

- The `data` parameter corresponds to the dataset. This parameter is an array of the form `[[Quantized; M], N]`, where `M` is the number of features and `N` is the number of samples.
- The `labels` parameter is an array of the form `[[Quantized; C], N]`, where `C` is the number of classes in the classification problem and `N` is again the number of samples. If the $i$-th sample belongs to class $c$ for $c \in \{0,...,C - 1\}$, then `labels[i][c] == Quantized { x: 65536 }` (which is the value of 1 represented in the field), and the other positions for `label[i]` should be equal to `Quantized { x: 0 }`.
- The `learning_rate_ratio` equals `learning_rate*ratio`, where `learning_rate` is the learning rate $\alpha$ that will be used during the training (assumed value between $[-0.1,0.1]$) and the `ratio` is the value of `1 / N`. This is left to the Prover since we want to save the computation of this quantity in the circuit directly. We assume you want to train with at least $N=10$, so this value will bebetween $[0,0.1]$. In the code the combined value `learning_rate_ratio` is asserted to have max 11 bits ($0.1*0.1=0.01$ has 11 bits).
- The `epochs` parameters is the number of epochs $E$ that will be used for the training.
- And parameters correspond to the parameters that the Prover will prove to.

The complete implementation and instructions can be found in the GitHub [repository of the ML library implemented in Noir](https://github.com/hashcloak/noir-mpc-ml).

