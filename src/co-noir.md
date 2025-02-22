## Co-Noir

The goal of this project was to be able to execute the training of the model with [co-noir](https://github.com/TaceoLabs/co-snarks/tree/main/co-noir/co-noir), a tool for creating collaborative SNARKs created by [Taceo](https://taceo.io/). This is what adds the special MPC sauce to the functionality, and we were lucky enough that Taceo had been doing the heavy lifting of creating a prover in MPC. This tool was relatively new and they were adding necessary features during the process to make this library practical, about which we will share a bit in this section. Furthermore, we've added the steps you can take to run the ml functionality with co-noir yourself. 

### Dancing with co-noir

After our initial rounds of optimization, we found major gatecount improvement by using unconstrained functions (these are functions that do not require constraints to be enforced in the ZK circuit). However, brillig (unconstrained) code was not yet supported by co-noir at that point. We communicated this to the Taceo team and only 10 days later, they had already added enough features to their brillig VM to support all of our optimizations that used unconstrained code :rocket:

Another challenge presented itself when the full code was indeed supported by the co-noir tooling, but we were trying to make the tests heavier and heavier by iterating for more epochs; up until now we had been testing with one epoch. First after increasing to 5 epochs, the network was getting overloaded by the increasing size of the circuits and the network interface trying to send it all at once. When that was fixed we got bold and tried to increase from 5 to 10 epochs. This time, the process got killed by RAM. To battle the RAM spikes, within two days the Taceo team implemented a better suited MPC-sorting algorithm, which made way to running the training algorithm for 10 and even 20 epochs.

### CMUX optimization

During our communications with the Taceo team, they gave us another insight to optimize our code, namely by using conditional multiplexer (CMUX) friendly code structures. The form of operation

$$
x=c⋅(a−b)+b
$$

is a bit more friendly to ZK/MPC operations and in typical code can look like this `x = if c {a} else {b}`. 

Following this tip, we changed the code for scaling down a value by the factor. This is being done after a multiplication or a dot product requires to flip the sign if we're working with negative values, as explained [above](#Multiplication) in the section about fixed point values. This is an old snippet, in which byte decomposition was done twice (because in ZK all branches are always executed):

```=rust
// ...
let mut temp: Field = self.x * other.x;

// byte decomposition, needed for scaling down
let mut bytes: [u8; 32] = temp.to_be_bytes();

let negative = is_negative(temp);

// In case of a negative value, we flip the sign
// if that is the case, the byte decomposition has to be redone
if negative {
    temp = 21888242871839275222246405745257275088548364400416034343698204186575808495616
        - temp
        + 1;
    bytes = temp.to_be_bytes();
}

// Scaling down
// Flip sign back
// Return result
```

We first changed the return value of the function `is_negative` from a boolean to `Field`, where 0 represents false and 1 true. This can be decided checking whether the value used the lower or higher bits of the register (lower bits = positive, higher bits = negative). This was the code before:

```rust=
pub fn is_negative(x: Field) -> bool {
    let (_, higher_bytes) = decompose(x);
    higher_bytes != 0
}
```

We present next the new version of the `is_negative` function:

```rust=
// returns 1 for true, 0 for false
pub fn is_negative(x: Field) -> Field {
    let (_, higher_bytes) = decompose(x);
    if higher_bytes == 0 {
        0
    } else {
        1
    }
}
```

Since the `is_negative` returns 0 or 1, we could actually use the CMUX operation directly to optimize one byte decomposition away:

```=rust
// ...
let mut temp: Field = self.x * other.x;
let negative = is_negative(temp);

// Calculate the temp value
temp = negative
    * (
        21888242871839275222246405745257275088548364400416034343698204186575808495616 
        - temp
        + 1
        - temp
    )
    + temp;

// Perform byte decomposition only once
let bytes: [u8; 32] = temp.to_be_bytes();
```

### Run `ml` functionality with co-noir

To execute the functionality from the `ml` library with co-noir, you can use the skeleton project [here](https://github.com/hashcloak/noir-mpc-ml/tree/master/benchmarking/conoir_project). This includes necessary configurations for the three parties, their respective keys and certificates, and the generators for the BN254 curve. Also, it contains a [script](https://github.com/hashcloak/noir-mpc-ml/blob/master/benchmarking/conoir_project/run_full_mpc_flow.sh) which executes the complete MPC-ZK flow:

1. Split input into shares
2. Witness generation in MPC
3. Proving in MPC
4. Creation of verification key
5. Verification of proof

Additional examples of a flow like this can be found in the examples [folder](https://github.com/TaceoLabs/co-snarks/tree/main/co-noir/co-noir/examples) of co-noir itself. 

Then, in the [`main.nr`](https://github.com/hashcloak/noir-mpc-ml/blob/master/benchmarking/conoir_project/src/main.nr) you can add the desired functionality. Make sure to either use a local version of the library or replace the `Nargo.toml` import with the correct one. Example code:
```=rust
use noir_mpc_ml::ml::train_multi_class;
use noir_mpc_ml::quantized::Quantized;

fn main(inputs: [[Quantized; 4]; 30], labels: [[Quantized; 30]; 3]) -> pub [([Quantized; 4], Quantized); 3] {
    let learning_rate = 6554;
    let ratio = 2185;

    let epochs = 10;

    let learning_rate = Quantized::new(learning_rate);
    let ratio = Quantized::new(ratio);
    let parameters = train_multi_class(epochs, inputs, labels, learning_rate, ratio);
    parameters
}
```
