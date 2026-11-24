# I built tiny NumPy and learned why NumPy is fast

This week I wrote **tiny-vec**: vectors, matrices, Strassen multiplication, SVD,
PCA and a reverse-mode autograd engine, all in pure Python with zero
dependencies. NumPy was allowed in exactly one place: the tests, as the oracle
my code had to agree with.

**What I built.** A `Vector` with dot products and norms; a `Matrix` with
transpose and two matmuls (the triple loop and Strassen); a scalar `Value` that
records how it was computed so `backward()` can walk the graph in reverse
topological order and apply the chain rule; a small MLP on top of it that
learns XOR in 200 steps; and an SVD that never calls a library SVD. It
power-iterates on AᵀA, then deflates, one singular value at a time. My PCA on
iris matches scikit-learn's to 1e-4, up to the sign of each axis.

**The speed gap.** A 128×128 matmul took 95 ms in my code and 0.43 ms in NumPy,
about 220× slower. Measuring it made the reason concrete. Each of my ~2 million
multiply-adds is a bytecode dispatch on two boxed Python floats. NumPy hands the
whole loop to BLAS: contiguous memory, SIMD lanes, cache-blocked tiles and
several cores, with no interpreter anywhere in the inner loop. The algorithm is
the same; nearly all of the difference is constant-factor overhead.

**The Strassen surprise.** Strassen replaces 8 recursive multiplies with 7 and
gets O(n^2.807). Textbooks put the crossover with the naive method around
n=128. On my machine, in Python, 128 was still a clear win for naive, 256 was a
coin-flip that changed from run to run, and Strassen only won clearly at 512
(5.7 s vs 9.4 s). The missing multiply is paid for with 18 extra matrix
additions per level, and in pure Python additions aren't cheap. Big-O tells you
which way the curves bend. Only a benchmark tells you where they cross.

**What stuck.** Gradients accumulate with `+=` because a value used twice gets
gradient from both paths. Writing the numeric-gradient test caught that mistake
faster than reading the code did. Property tests (hypothesis) and oracle tests
(NumPy) made it safe to move fast.

Code, benchmarks and plots: see the [README](../README.md).
