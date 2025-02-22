## Fixed-point arithmetic

In machine learning, it is common to use decimal values in the datasets. For example, the [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) contains 151 data samples with features of flowers and to which of the 3 species options they belong: "setosa", "versicolor" or "virginica". This is a snippet with some of the entries:

| sepal_length | sepal_width | petal_length | petal_width | species   |  
|--------------|-------------|--------------|-------------|-----------|  
| 5.1          | 3.5         | 1.4          | 0.2         | setosa    |  
| 7.0          | 3.2         | 4.7          | 1.4         | versicolor|  
| 7.7          | 2.8         | 6.7          | 2.0         | virginica |  
| 6.7          | 3.3         | 5.7          | 2.1         | virginica |  


The problem here is that the standard Noir data types do not support decimal values, so one of the first tasks was to decide how to represent these data points in our system. There were three things to consider:

- We need support for decimal values.
- We need support for negative values.
- We need to achieve good performance

After some testing, we decided to go with the strategy proposed by [Catrina and Saxena](https://www.ifca.ai/pub/fc10/31_47.pdf), in which decimal values can be represented as fixed-point values, and the latter can be encoded as elements in a finite field. Following that idea, we store a signed and scaled fixed-point value in a single `Field` element:

```=rust
pub struct Quantized {
    pub x: Field,
}
```

- **Dealing with signed values:** if the value is positive, it is represented as itself. If it is negative, it is represented as `p - x` where `p` is the modulus of the field we're working over. 
- **Bitsizes:** A field element has 254 bits where the first half can be used for positive values, and the second half for negative ones. However, we want to make use of the function `decomposed`, which splits a Field up into 128 first bits and the remaining 126 last bits. So for both positive and negative values we will use a maximum of 126 bits, and the two "left over" bits in the middle should stay unused always.
- **Prevent overflow/underflow:** to maintain above bitsize always, we add checks on the bitsize of inputs before performing arithmetic. More on this later on. 
- **Scaling the values:** all values we work with have been multiplied by $2^{16}$. This means they are scaled by $2^{-16}$. For example, `Quantized { x: 65536 }` equals 1, and `Quantized { x: p-1 }` equals $-2^{-16}$. 

Visually, you can think of a Field element with space for 254 bits like this `|_,_,...,_,_|`. We can represent both positive and negative numbers in this type, by occupying different areas within that field. Positive numbers will only have maximum 126 bits set in the first half: `|x,x,x,x,..,x, .. ,_,_,_,_|`. Negative numbers will only have maximum 126 bits set in the second half: `|_,_,_,_, .. x,x,x,..,x|`. We prevent overlap that can occur due to arithmetics by asserting the proper bitsizes on inputs before doing any addition, subtraction or multiplication. The benefit of this choice is that a Field is already the standard type of Noir; all other types such as u8, u16, u32 etc. under the hood are Fields with additional rangechecks. This means that Fields are the most performant, which is exactly what we are looking for. 

### Simple arithmetic 

Another benefit of this approach is that arithmetic is rather straightforward, because of the modular arithmetic in a field. Adding a pair of `Quantized` values automatically works by adding the underlying field values. The same holds for subtraction. For example, let's say we have decimal values $4.5 + (-3.1)$. First we represent them as `Quantized`:

- $4.5*2^{16} = 294912$
- $3.1*2^{16} = 203161.6$. Notice that this value has some decimal numbers, so we truncate it to $203161$.

In Noir we can write this as follows:

```rust
let a: Field = 294912;
let a_quantized = Quantized { x: a };
let b: Field = -203161; // This automatically becomes p-203161
let b_quantized = Quantized { x: b };
```

Then, we add them as follows:

```rust
let addition1_quantized = a1_quantized + b1_quantized;
````

The above operation is equal to 

$$
294912 + (p-203161) = 294912-203161 + p = 91751 + p,
$$

which is equals to $91751$ modulo $p$. 

Now, if we calculate the decimal representation of the result (i.e. converting back from `Quantized` to decimal value), we obtain the following result:

$$
91751* \frac{1}{2^{16}} = 1.400,
$$

and if we simply compute $4.5 + (-3.1)$, we see that $1.4$ is indeed the expected value. 


### Multiplication

The most difficult arithmetic operation is multiplication due to the scale we are using. When we naively multiply 2 values, the result will be an order of scale too large. Let's say we are working with $a*2^{16}$ and $b*2^{16}$, then a straightforward multiplication results in $a*b*2^{32}$. Division by the original scale is needed to get the result right. This means we have to do a multiplication and then scale down by dividing by $2^{16}$. How to approach this?

Our initial idea was to convert our Field element to bytes and then scale down by simply truncating it by 2 bytes. Since the scale is $2^{16}$, this equals 2 bytes. Afterwards, just convert the leftover bytes to a Field. That would look something like this (pseudocode):

```rust=
pub struct Quantized {
    pub x: Field,
}

pub fn chop_off_two_bytes(x: Field) -> Field {
    let bytes: [u8; 32] = x.to_be_bytes();
    let mut truncated = [0; 32];
    for i in 0..30 {
        truncated[i + 2] = bytes[i];
    }
    Field::from_be_bytes::<32>(truncated)
}

let res = a.x * b.x;
let scaled_down = chop_off_two_bytes(res);
```

But wait! What if our initial multiplication is a negative value? Then this whole approach doesn't make sense, because the bits are positioned in "reverse" order. To address this, in the case that we're working with a negative value, temporarily switch to a positive version of that value to do the scaling, and then switch back to negative. This is the idea:

```rust=
let mut res = a.x * b.x;
let negative = is_negative(res);
if negative {
    res = p - res;
}
let mut scaled_down = chop_off_two_bytes(res);
if negative {
    scaled_down = p - scaled_down;
}
```

It turns out that there is a more performant way to do the flipping of the sign, which will be discussed later on. 

To improve the division approach, we ended up incorporating a suggestion by Tom French of the Aztec team (sign flipping left out for clarity): 

```rust=
let res = a.x * b.x;

// Cast x to a u16 to preserve only the lowest 16 bits.
let lowest_16_bits = res as u16;

// Subtract off the lowest 16 bits so they are cleared.
let res_with_cleared_lower_bits = res - lowest_16_bits as Field;

// The lowest 16 bits are clear, `res_with_cleared_lower_bits` is divisible by `65536`,
// therefore field division is equivalent to integer division.
let final_res: Field = res_with_cleared_lower_bits / 65536;
```
### Restrictions on bitsizes

As described above, a positive value will have maximum 126 bits in the first half of the Field element, while a negative value has a maximum of 126 bits at the end of the 254 bits of a Field. To maintain this invariant, we must assert limits on the bitsizes before performing arithmetic operations:
- multiplication: both inputs can have max 63 bits. Then the result can become 63+63=126 bits.
- addition and subtraction: both inputs can have max 125 bits. Then the result can become 125 + 1 carry bit = 126 bits.
- division($a,b$): $a$ can have 109 bits and $b$ can have 126 bits. We'll explain this next.

For division we have to keep in mind that the result must have the correct scale, just like with multiplication. However in this case, a naive division would lead to a scaled down result. To prevent this, we scale the numerator up by $2^{16}$ before performing the division. Since the scale itself is 17 bits, this means the initial numerator input must limited to have 126-17=109 bits. The denominator can have 126 bits. 

In the standalone library for the fixed point arithmetic, these checks have been added to the respective functions. For the ML library however, we opted to use a "stripped" version of quantized directly in the library itself, which doesn't contain any bitsize checks. Instead, the bitsize checks have been added throughout all of the ML functionality iself. This was done to optimize the amount of bitsize checks needed by tailoring them to the actual functionality. Throughout [the code](https://github.com/hashcloak/noir-mpc-ml/blob/master/src/ml.nr#L126) the bitsize checks can be found as well as the reasoning for it. 


### Use the fixed point library

The quantized functionality can be found and used in a separate library called [noir-fixed-point](https://github.com/hashcloak/noir-fixed-point). Here, we present some instructions on how to use it. 

First, you need to add the library as a dependency in your own Noir project:

```=toml
[dependencies]
fixed_point = { tag = "v0.1.2", git = "https://github.com/hashcloak/noir-fixed-point" }
```
    
Then, you can create values and perform arithmetic as it is shown in the following example:

```rust
let a = Quantized::new(98304); // Represents 1.5
let b = Quantized::new(147456); // Represents 2.25

let add = a + b; // (98304 + 147456) / 2^16 = 3.75
let sub = a - b; // (98304 - 147456) / 2^16 = -0.75
let mul = a * b; // (98304 * 147456) / 2^(16 + 16) = 3.375
```
