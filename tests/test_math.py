import math
import numpy as np
from tiny_vec import Value, MLP


def test_engine_matches_numeric_gradient():
    def f(a, b):
        return (a * b + a ** 2) * b.relu()
    a, b = Value(-1.5), Value(2.0)
    out = f(a, b); out.backward()
    h = 1e-6
    ga = (f(Value(-1.5+h), Value(2.0)).data - f(Value(-1.5-h), Value(2.0)).data)/(2*h)
    assert math.isclose(a.grad, ga, abs_tol=1e-4)


def _numeric_grad(f, xs, i, h=1e-6):
    up = [x + (h if j == i else 0.0) for j, x in enumerate(xs)]
    dn = [x - (h if j == i else 0.0) for j, x in enumerate(xs)]
    return (f(*map(Value, up)).data - f(*map(Value, dn)).data) / (2 * h)


def test_every_op_matches_numeric_gradient():
    # exercises + - * / ** neg tanh exp relu and the reflected (r*) operators
    def f(a, b):
        return ((2 - a) / b + 1 / (a * b) - (-b).exp() + (a - 3) * b.tanh()
                + 0.5 * a ** 3 + (a + 1).relu()) / 4
    xs = [0.7, -1.3]
    a, b = Value(xs[0]), Value(xs[1])
    f(a, b).backward()
    assert math.isclose(a.grad, _numeric_grad(f, xs, 0), abs_tol=1e-4)
    assert math.isclose(b.grad, _numeric_grad(f, xs, 1), abs_tol=1e-4)


def test_mlp_shapes_and_zero_grad():
    model = MLP(3, [4, 2])
    assert len(model.parameters()) == (3 * 4 + 4) + (4 * 2 + 2)
    out = model([Value(1.0), Value(-1.0), Value(0.5)])
    assert len(out) == 2
    sum(out, Value(0.0)).backward()
    assert any(p.grad != 0.0 for p in model.parameters())
    model.zero_grad()
    assert all(p.grad == 0.0 for p in model.parameters())
