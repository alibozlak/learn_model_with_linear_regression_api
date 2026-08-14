# learn_model_with_linear_regression_api

Multivariable linear regression trained with batch gradient descent, exposed over
HTTP. Written from scratch — no `ndarray`, no `linfa`, no autodiff — so that every
step of the math is visible in the source.

The "without feature scaling" in the module name is deliberate: the features keep
their original ranges, which makes the cost surface a long narrow valley and forces
a very small learning rate. Seeing that behaviour is the point of this project, not
a limitation to be worked around.

## The math

The hypothesis is a plain linear combination of the `n` features plus a bias:

```text
f(x) = a_1 * x_1 + a_2 * x_2 + ... + a_n * x_n + b
```

The cost being minimised is the mean squared error over the `m` samples:

```text
J = (1/m) * sum over i of ( f(x^(i)) - y^(i) )^2
```

There is no `1/2` in front of the mean, so the partial derivatives keep their factor
of 2:

```text
dJ/da_j = (2/m) * sum over i of ( x_j^(i) * ( f(x^(i)) - y^(i) ) )
dJ/db   = (2/m) * sum over i of ( f(x^(i)) - y^(i) )
```

Each iteration of batch gradient descent updates every parameter once, using
derivatives computed from the *same* unchanged coefficients — the update is
simultaneous, which is why the new values are staged in a temporary vector before
being written back.

## Requirements

- Rust (edition 2024)
- No system dependencies

## Running

```bash
cargo run
```

The server binds `0.0.0.0:3000` and prints the address it is listening on. Use
`cargo run --release` for anything larger than a toy data set — the training loop is
roughly two orders of magnitude faster with optimisations on.

## API

### `POST /train`

Trains a fresh model on the data set in the request body and returns the
coefficients it converged to. Each request starts from all-zero coefficients; the
server keeps no state between requests.

**Request body** — raw JSON. `inputs[i]` is one sample's feature values, and
`outputs[i]` is that sample's expected value:

```json
{
  "inputs": [
    [55.0,  1.0],
    [72.0,  2.0],
    [90.0,  3.0],
    [110.0, 3.0],
    [130.0, 4.0],
    [165.0, 5.0]
  ],
  "outputs": [23000.0, 30500.0, 38000.0, 44000.0, 48500.0, 61000.0]
}
```

Every sample must carry the same number of features. The feature count `n` is read
from the first sample.

**Response** — `200 OK`, single-line JSON:

```json
{"last_coefficients":[375.13,-195.00,1807.28],"J_before_learning":1543200000.0,"J_after_learning":2847361.5}
```

`last_coefficients` is laid out as `[a_1, ..., a_n, b]` — the bias occupies the last
slot, so the array is `n + 1` long. `J_before_learning` and `J_after_learning` are
the cost before and after the run; comparing them is the quickest way to tell
whether the learning rate was sane.

**Errors** — `4xx`/`5xx` with a plain-text body describing the problem:

| Status | Body | Cause |
| --- | --- | --- |
| `400` | `invalid JSON payload: ...` | Body is not valid JSON, or does not match the expected shape |
| `400` | `"inputs" is empty, there is nothing to train on` | `inputs` is `[]` |
| `400` | `sample count mismatch: N input sample(s) but M output(s)` | `inputs` and `outputs` differ in length |
| `400` | `sample K has F feature(s) while the first sample has E` | Rows of differing width |
| `500` | serialisation error | The result could not be encoded |

## Usage with curl

Send a data set held in a file. `--data-binary` matters — plain `-d` strips newlines
and can mangle the payload:

```bash
curl -X POST --data-binary @dataset.json http://localhost:3000/train
```

Inline, for a quick check:

```bash
curl -X POST --data-binary '{"inputs":[[55.0,1.0],[130.0,4.0]],"outputs":[23000.0,48500.0]}' http://localhost:3000/train
```

No `Content-Type` header is needed: the handler takes the body as a raw string and
parses it itself, so curl's default header is not rejected.

Pretty-print the response by piping through `jq`:

```bash
curl -s -X POST --data-binary @dataset.json http://localhost:3000/train | jq
```

## Layout

```text
src/
├── main.rs                            HTTP layer: routing, handler, server bootstrap
├── json_converter.rs                  JSON <-> the model's vectors, plus validation
└── learning_without_feature_scaling.rs   The model: hypothesis, cost, gradients, descent
```

The layering is one-directional: the model knows nothing about JSON, and
`json_converter` knows nothing about HTTP. `json_converter` validates the payload
before the model's constructor is reached, so data arriving from outside can never
trigger the panics that constructor documents.

## Known limitations

These are either deliberate or on the roadmap — listed so nobody has to rediscover
them.

- **Hyperparameters are hardcoded.** The learning rate (`0.000003`) and iteration
  count (`1_000_000`) live in the handler. Both should come from the request.
- **No divergence guard.** If the learning rate is too large for the data's scale,
  the coefficients overflow to `NaN`. `serde_json` writes non-finite floats as
  `null`, so the endpoint answers `200 OK` with
  `{"last_coefficients":[null,null,null], ...}`. Compare `J_after_learning` against
  `J_before_learning` to catch this until it is handled server-side.
- **Request bodies are capped at 2 MB** by axum's default body limit. Larger data
  sets are rejected with `413 Payload Too Large`; raising it needs a
  `DefaultBodyLimit` layer.
- **Training blocks an async worker thread.** The descent is synchronous CPU work
  running directly in the handler; it belongs in `tokio::task::spawn_blocking`.
- **The gradient loop recomputes predictions per parameter.** `dJ/da_j` calls the
  hypothesis for every sample, and it is called once per feature, making an
  iteration `O(m * n^2)` where `O(m * n)` would do. Caching the per-sample errors
  once per iteration fixes it.
- **No feature scaling.** By design — see the top of this file.
