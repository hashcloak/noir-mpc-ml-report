## Optimizations

Once we had the basic functionality for training a model done, we decided to focus on optimizations. This is dangerous territory, as we would discover, but in many cases there are quite a few things you can do to optimize your code and bring the gatecount down. There are two main steps to take here: find opportunities for optimization and then design the actual optimizations itself. We'll touch on those in this section. Lastly, we'll also share how optimizations can lead to underconstraining your circuit and our lessons learned here. 

### How to find opportunities for optimizations

Somewhere, far far away in a Noir Discord thread, we noticed the mention of a "profiler" that is available in the Noir Github. Notably, the Aztec team itself uses it to profile large circuits. With a profiler, you are able to identify costly parts of your code and thus get direction on where to focus optimization efforts. We definitely wanted to give this tool a try! The only downside; it came with this warning "This profiler is not yet documented or advertised publicly as it is still slightly in flux and used only by the Aztec team." Luckily, we like a challenge. 

As it turns out, using the profiler is totally doable. You can find the profiler tool [here](https://github.com/noir-lang/noir/tree/master/tooling/profiler). These are the steps we took to make it work:
1. Git clone the Noir repo
2. Checkout the tag of the nargo version you intend to use. Make sure the installed `bb` version aligns with it.
3. Set `PATH_TO_ARTIFACT`to where the `json` of your compiled circuit is located, for example `target/test.json`.
4. Set `PATH_TO_OUTPUT` to where the `svg` file will be saved, for example `/output`.
6. Set `PATH_TO_BB` to where `bb` is installed.
7. Obtain gates flamegraph: 
```=bash
cargo run --package noir_profiler --bin noir-profiler  gates-flamegraph --artifact-path PATH_TO_ARTIFACT --output PATH_TO_OUTPUT --backend-path PATH_TO_BB
```
10. Obtain opcodes flamegraph 
```=bash
cargo run --package noir_profiler --bin noir-profiler  opcodes-flamegraph --artifact-path PATH_TO_ARTIFACT --output PATH_TO_OUTPUT
```

The one we focused on is the gates flamegraph. The first iterations looked something like this:
![Screenshot 2025-01-16 at 11.55.02](https://hackmd.io/_uploads/rk3GdTID1l.png)

We highly recommend the use of this profiler, and have also mentioned this to the Aztec/Noir team. Apart from helping you steer your optimization efforts in a certain direction, it also helps developing intuition for the cost of certain things in circuits (in Noir). Below we will discuss some of the optimizations we implemented. 

### Unnecessary re-calculations

As developers, we often rely on compilers to optimize code for us, and we don't always look at a "simple" multiplication more or less. However, in the world of ZKP that can be costly. In this case it cost us 127,000 gates. Let us explain.

As discussed above, the `sigmoid` function is an important building block for the training function. In this implementation, it does an approximation instead of the sigmoid function itself. In this approximation, we decide in which of the 5 cases we find ourselves and based on that return one of the following outputs:
```rust=
let outputs = [
    Quantized::new(6), // 0.0001
    (Quantized::new(1819) * x) + Quantized::new(9502), // 0.02776 and 0.145
    (Quantized::new(11141) * x) + Quantized::new(32768), // 0.17 and 0.5
    (Quantized::new(1819) * x) + Quantized::new(56031), // 0.02776 and 0.85498
    Quantized::new(65529), // 0.9999
];
```

The culprit was `(Quantized::new(1819) * x)`, which appears twice. Of course we could have seen that directly in the code, but it stood out to us from the flamegraph. We might have assumed this would be optimized away somehow, but that was not the case. The fix was simple; assign the value to a variable and use it in both spots. This brought down the gatecount about ~127K gates at the time that we were benchmarking this. 

### Optimization dot product

The flamegraph revealed what the expensive operations in our code were: Quantized multiplication, and in particular scaling down after doing the multiplication. 

After a multiplication strictly speaking it is needed to scale down to get the right answer. But what if you don't need the "right answer" right away? This was the case for the dot product in the code for obtaining the prediction:
```rust=
fn get_prediction<let M: u32>(
    weights: [Quantized; M],
    inputs: [Quantized; M],
    bias: Quantized,
) -> Quantized {
    let mut z = Quantized::zero();
    for i in 0..M {
        z += weights[i] * inputs[i];
    }

    approx_sigmoid(z + bias)
}
```
At line 8 for each iteration a multiplication with a scale down happens, before adding all the resulting values together. In addition to the expensive scaling down operation, in each quantized multiplication 2 bitsize checks are done in order to prevent overflow during the multiplication ([code ref](https://github.com/hashcloak/noir-fixed-point/blob/added_checks/src/quantized.nr#L112)). Instead of using this straight forward dot product, we customized it to scale down fewer times and adjust the bitsize checks to fit within our larger functionality and be cheaper. 

First, let's look at delayed scaling down. The opportunity at line 8 is to realize that we can perfectly add multiple scaled up values (which happens after naively multiplying the quantized values and not scale down) and scale down just once after all additions have taken place. Since scaling down is so expensive, this seems like a good idea. Here, we're not yet taking into account the possible overflow, which we'll talk about next. The intermediate solution would look something like this:

```rust=
fn get_prediction<let M: u32>(
    weights: [Quantized; M],
    inputs: [Quantized; M],
    bias: Quantized,
) -> Quantized {
    // let z = weights dot_product inputs
    let mut z = 0;
    for i in 0..M {
        // Perform operations directly on Field elements, scale down at the end
        z += weights[i].x * inputs[i].x;
    }

    // Scale the intermediate value down due to multiplications and add bias
    approx_sigmoid(Quantized { x: scale_down(z) } + bias)
}
```

The number of scaling down operations is brought down from `M` to 1. In addition to that, we're doing Field multiplications instead of Quantized multiplications which is cheaper, since we're avoiding the previously mentioned bitsize checks in quantized multiplication. However, we're in risk of overflow. 

We need to bring some bitsize checks back to make sure no intermediate overflow is happening. Recall that `inputs` are being restricted to 20 bits from the start, so we don't need to do additional bit checks on those. We just need to make sure that the `weights` are max 105 bits and the intermediate `z` values don't go over 125 bits. In this way, the multiplication on the right hand side becomes maximum 125 bits, and their addition 126 bits, falling exactly within the necessary bounds:
```rust=
assert_bitsize::<105>(weights[i]);
assert_bitsize::<125>(Quantized { x: z });
z += weights[i].x * inputs[i].x;
```

The same type of optimization has been applied in the `train` function ([ref](https://github.com/hashcloak/noir-mpc-ml/blob/master/src/ml.nr#L71)). 

### A surprising optimization

During our quest to squeeze out at the gates we possibly could, we encountered the following optimization that we only came to understand with help from the Noir team. Before:

```rust=
let mut index = 4;
if x <= cuts[0] {
    index = 0;
} else if x <= cuts[1] {
    index = 1;
} else if x <= cuts[2] {
    index = 2;
} else if x <= cuts[3] {
    index = 3;
}
outputs[index]
```

After (~3k gates cheaper at the time):

```rust=
let mut res = outputs[4]; // Default to the last index in case x is above all cuts
// Determine the correct interval index by checking against each cut
if x <= cuts[0] {
    res = outputs[0];
} else if x <= cuts[1] {
    res = outputs[1];
} else if x <= cuts[2] {
    res = outputs[2];
} else if x <= cuts[3] {
    res = outputs[3];
}
res
```

Note that the `outputs` array is already known beforehand in both cases:
```rust
let outputs = [
    Quantized::new(6), // 0.0001, 0.0001 / 2^-16 = 6.5536
    temp + Quantized::new(9502), //0.02776 and 0.145, 0.02776 / 2^-16 = 1819.27936, 0.145/2^-16 = 9502.72
    (Quantized::new(11141) * x) + Quantized::new(32768), //0.17 and 0.5, 0.17 / 2^-16 = 11141.12, 0.5/2^-16 = 32768
    temp + Quantized::new(56031), //0.02776 and 0.85498, 0.85498/2^-16 = 56031.96928
    Quantized::new(65529), //0.9999 / 2^-16 = 65529.4464
];
```

@[TomAFrench](https://github.com/TomAFrench) of the Aztec team pointed out that because the array content is known at compile time, at runtime only the correct one has to be selected in the optimized code snippet. This is called value merging. 

In the earlier version of the code, at compile time we don't know which entry of `outputs` we're going to read and this causes the value merging to happen on the index we'll be reading from. This, Tom explained to us, forces the generation of a memory block and do a dynamic read from array, which is more expensive. 

The difference is that in the optimized code, we could replace array access with the actual value, whereas in the initial code, we can't do that. A good reminder that everything known at compile-time will be cheaper within circuits, and reasoning how to do the least amount of work at runtime will help us optimize the code. 

### Lesson learned: optimization or underconstraining?

One of the early optimizations we tried to introduce was for the byte decomposition of a Field. If you recall the section on multiplication for fixed point arithmetic above, we started out scaling down after multiplication using a trick to decompose the Field into bytes and then chop off 2 of them, resulting in a division by $2^{16}$. It so happens that byte decomposition is quite expensive in circuits, and our code used a lot of multiplications and thus this scaling down trick. When studying the flamegraph, we knew: this is where we can make a big difference!

After browsing through the Noir documentation, we found an interesting [code snippet](https://noir-lang.org/docs/noir/concepts/data_types/fields#to_be_bytes):

```rust=
fn test_to_be_bytes() {
    let field = 2;
    let bytes: [u8; 8] = field.to_be_bytes();
    assert_eq(bytes, [0, 0, 0, 0, 0, 0, 0, 2]);
    assert_eq(Field::from_be_bytes::<8>(bytes), field);
}
```

The Field type has both a function to decompose into bytes, `to_be_bytes`, and one to create a Field from bytes, `from_be_bytes`. What if we performed the first action in an unconstrained function and used the second one to constrain the actual byte decomposition? It seemed worth a try, and since we were learning about the optimizations for Noir we were not sure of the outcome. This is the code we used (**do not use it!**):
```rust=
unsafe {
    bytes = get_bytes(temp);
}

assert(Field::from_be_bytes::<32>(bytes) == temp);
```

To our surprise, this indeed led to a gate count improvement.

Unfortunately, this is exactly when we introduced a security issue. 

Once again, @[TomAFrench](https://github.com/TomAFrench) from the Noir team helped us understand and pointed out this issue. Basically, our 'optimization' took a shortcut compared to the [byte decomposition function](https://github.com/noir-lang/noir/blob/c44b62615f1c8ee657eedd82f2b80e2ec76c9078/noir_stdlib/src/field/mod.nr#L152-L187) from the standard library. The check we bypassed ensured that the value that the byte array represented fell within the Field. In our implementation, $n$ would be equal to $p+n$ for $p$ the modulus of the field, while in the correct implementation an additional check is added to prevent that. In conclusion, our circuit became underconstrained due to our optimization efforts.

To fix the problem, we had to use the the (constrained) version of `to_be_bytes` that we started out with. In the end, the code became obsolete because we used the improved scaling down code that Tom proposed, which doesn't use byte decomposition. However, for the future we are certainly warned and understand better the workings of these functions. 

This was a harsh lesson to learn, and we hope our experience helps you avoid making the same mistake.