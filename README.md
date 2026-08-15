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

Docker alone is enough if you would rather not install a toolchain — see
[Docker](#docker) below.

## Running

```bash
cargo run
```

The server binds `0.0.0.0:3000` and prints the address it is listening on. Use
`cargo run --release` for anything larger than a toy data set — the training loop is
roughly two orders of magnitude faster with optimisations on.

## Docker

A multi-stage `Dockerfile` is included, so building and running takes nothing but
Docker itself:

```bash
docker build -t linear-regression-api .
```

```bash
docker run --rm -p 3000:3000 linear-regression-api
```

The image is always the optimised build, so the `--release` caveat above does not
apply to it.

How it is put together, and why:

- **Two stages.** `rust:1.95-slim-bookworm` compiles and `debian:bookworm-slim` runs.
  Only the binary crosses between them, which lands the result at roughly 77 MB
  rather than the gigabyte-and-a-half the toolchain image weighs. Both stages sit on
  the same Debian release, so the binary meets the glibc it was linked against.
- **Dependencies are cached on a layer of their own.** The manifests are copied and
  built against a placeholder `main.rs` before the real sources arrive, so editing
  `src/` recompiles this crate alone and leaves axum, tokio and everything under them
  untouched.
- **It runs as an unprivileged user** (`app`, uid 999). The server opens no files and
  binds an unprivileged port, so there is nothing for root to do.
- **The port is fixed at 3000 inside the container**, because `main.rs` binds
  `0.0.0.0:3000` unconditionally — there is no environment variable to change it.
  Remap it from the host instead if that port is taken: `-p 8080:3000`.

`.dockerignore` keeps `target/` out of the build context. It grows to a few hundred
megabytes as soon as the project is built locally, and every byte would otherwise be
handed to the daemon on each build.

## API

### `POST /train`

Trains a model on the data set in the request and returns the coefficients it
converged to. The server keeps no state between requests — the point the descent
starts from arrives with the request, as `initial_coefficients`.

**Request** — `multipart/form-data` carrying a single field named `dataset`, whose
content is the training set as JSON. `inputs[i]` is one sample's feature values,
`outputs[i]` is that sample's expected value, and `initial_coefficients` is where
gradient descent begins:

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
  "outputs": [23000.0, 30500.0, 38000.0, 44000.0, 48500.0, 61000.0],
  "initial_coefficients": [0.0, 0.0, 0.0]
}
```

Every sample must carry the same number of features. The feature count `n` is read
from the first sample.

`initial_coefficients` is **required** — leaving it out is a `400`, not a fall back
to zeros. It is laid out as `[a_1, ..., a_n, b]` with the bias last, so it must be
exactly `n + 1` long: three entries for the two-feature set above.

**Not wanting to choose a starting point still means sending the field**, filled with
`n + 1` zeros. There is no shorter stand-in and no value that means "decide for me" —
a zero vector of some other length is rejected by the length check exactly like any
other wrong length, so the number of zeros has to follow `n`:

| Features `n` | Send as `initial_coefficients` |
| --- | --- |
| 1 | `[0.0, 0.0]` |
| 2 | `[0.0, 0.0, 0.0]` |
| 3 | `[0.0, 0.0, 0.0, 0.0]` |

`[0.0, 0.0]` against a two-feature set is a `400`, not a request to start from the
origin — the two zeros are one short of the three that set needs.

That layout is deliberately the same one `last_coefficients` comes back in, which is
what makes a run resumable: feed a previous response's coefficients back in and the
descent picks up where it stopped. Splitting a run in two this way reaches exactly the
coefficients the undivided run would have — see [Resuming a run](#resuming-a-run).

**Query parameters** — both optional, both checked before any training starts:

| Parameter | Default | Accepted | Meaning |
| --- | --- | --- | --- |
| `learning_rate` | `0.000003` | finite and `> 0` | Step size of each gradient descent update |
| `loop_count` | `1000000` | `1` to `5000000` | How many iterations to run |

The defaults are values this kind of data set converges under. Because the features
are unscaled, a learning rate even one order of magnitude larger diverges — see the
note under [Known limitations](#known-limitations).

**Response** — `200 OK`, single-line JSON:

```json
{"last_coefficients":[386.94817672821796,-210.66512190655385,764.9043122781602],"J_before_learning":1818750000.0,"J_after_learning":4516124.886161309}
```

`last_coefficients` is laid out as `[a_1, ..., a_n, b]` — the bias occupies the last
slot, so the array is `n + 1` long, and it is accepted verbatim as the next request's
`initial_coefficients`. `J_before_learning` and `J_after_learning` are the cost before
and after the run; comparing them is the quickest way to tell whether the learning
rate was sane. Note that `J_before_learning` is the cost of the coefficients that were
*sent in*, so it is only the cost at the origin when zeros were sent.

**Errors** — `4xx`/`5xx` with a plain-text body describing the problem:

| Status | Body | Cause |
| --- | --- | --- |
| `400` | ``Invalid `boundary` for `multipart/form-data` request`` | The body was not sent as multipart — e.g. raw JSON via `--data-binary` |
| `400` | `Form hasn't a 'dataset' named field !!` | Multipart body carries no field called `dataset` |
| `400` | `Failed to deserialize query string: ...` | A query parameter is not of the expected type, e.g. `loop_count=abc` |
| `400` | `learning_rate must be a finite number greater than 0, got X` | `learning_rate` is zero, negative, `NaN` or infinite |
| `400` | `loop_count must be at least 1` | `loop_count` is `0` |
| `400` | `loop_count must not exceed 5000000, got X` | `loop_count` is above the cap |
| `400` | `invalid JSON payload: ...` | The field is not valid JSON, or does not match the expected shape |
| `400` | ``invalid JSON payload: missing field `initial_coefficients` at ...`` | The payload carries no `initial_coefficients` at all |
| `400` | `"inputs" is empty, there is nothing to train on` | `inputs` is `[]` |
| `400` | `sample count mismatch: N input sample(s) but M output(s)` | `inputs` and `outputs` differ in length |
| `400` | `sample K has F feature(s) while the first sample has E` | Rows of differing width |
| `400` | `"initial_coefficients count" - 1 is not equal to feature count !!` | `initial_coefficients` is not `n + 1` long — `[]`, and a zero vector too short for the set, included |
| `400` | ``Error parsing `multipart/form-data` request`` | The multipart body is malformed, or the request went over the 32 MB body limit |
| `500` | serialisation error | The result could not be encoded |

## Usage with curl

Send a data set held in a file. `-F` is what makes curl build the multipart body and
set the `Content-Type` boundary along with it; the `@` tells it to read a file rather
than take the text literally:

```bash
curl -X POST -F "dataset=@dataset.json" http://localhost:3000/train
```

Inline, for a quick check — no `@` this time, so the JSON is used as the value.
`initial_coefficients` has to be there even when it is all zeros:

```bash
curl -X POST -F 'dataset={"inputs":[[55.0,1.0],[130.0,4.0]],"outputs":[23000.0,48500.0],"initial_coefficients":[0.0,0.0,0.0]}' http://localhost:3000/train
```

Override either hyperparameter from the query string. Quote the URL, otherwise the
shell swallows everything from the `&` onwards:

```bash
curl -X POST -F "dataset=@dataset.json" "http://localhost:3000/train?learning_rate=0.000001&loop_count=2000000"
```

Pretty-print the response by piping through `jq`:

```bash
curl -s -X POST -F "dataset=@dataset.json" http://localhost:3000/train | jq
```

### Resuming a run

Because `last_coefficients` comes back in the layout `initial_coefficients` expects,
`jq` can splice one run's answer into the next request's payload:

```bash
curl -s -X POST -F "dataset=@dataset.json" "http://localhost:3000/train?loop_count=200000" | jq --slurpfile d dataset.json '$d[0] + {initial_coefficients: .last_coefficients}' > next.json
```

```bash
curl -s -X POST -F "dataset=@next.json" "http://localhost:3000/train?loop_count=200000"
```

Two runs of 200000 iterations on the sample data set land on exactly what a single run
of 400000 reaches — the same coefficients to the last bit, since nothing but the
starting point distinguishes the second call from a continuation of the first:

```text
0      -> 200000  {"last_coefficients":[386.94817672821796,-210.66512190655385,764.9043122781602],"J_before_learning":1818750000.0,"J_after_learning":4516124.886161309}
200000 -> 400000  {"last_coefficients":[385.6281263785745,-355.59201959840124,1418.405456447226],"J_before_learning":4516124.886161306,"J_after_learning":3767351.858096715}

0      -> 400000  {"last_coefficients":[385.6281263785745,-355.59201959840124,1418.405456447226],"J_before_learning":1818750000.0,"J_after_learning":3767351.858096715}
```

The one thing this buys that a longer `loop_count` does not is a look at the cost
part-way: each leg reports its own `J_before_learning` and `J_after_learning`, so a
run that is drifting towards divergence shows it before the whole budget is spent.

## Layout

```text
src/
├── main.rs                               Router, body limit, server bootstrap
├── train_endpoint.rs                     The POST /train handler
├── train_params.rs                       Query parameters and their validation
├── json_converter.rs                     JSON <-> the model's vectors, plus validation
└── learning_without_feature_scaling.rs   The model: hypothesis, cost, gradients, descent
```

The layering is one-directional: the model knows nothing about JSON, and
`json_converter` knows nothing about HTTP. `json_converter` validates the payload
before the model's constructor is reached, so data arriving from outside can never
trigger the panics that constructor documents. Its `n + 1` check on
`initial_coefficients` is what earns that guarantee now the vector comes from the
request rather than being built by the handler — the constructor panics on a
coefficient vector of the wrong length, and this is the check that stops one reaching
it.

## Known limitations

These are either deliberate or on the roadmap — listed so nobody has to rediscover
them.

- **A single-feature data set ignores `initial_coefficients`.** The handler replaces
  the vector with zeros whenever its length is 2 (`train_endpoint.rs:35`). Validation
  has already pinned the length to `n + 1` by that point, so the branch fires on
  exactly one thing: `n == 1`. A one-feature model therefore always starts at the
  origin, silently, whatever was sent — `initial_coefficients: [400.0, 1000.0]` on a
  one-feature set answers with `J_before_learning` measured at `[0.0, 0.0]`. The
  guard reads as though it were meant to catch an absent vector, but the field is
  required, so it can never fire for that reason. Every `n >= 2` set is unaffected.
- **No divergence guard.** If the learning rate is too large for the data's scale,
  the coefficients overflow to `NaN`. `serde_json` writes non-finite floats as
  `null`, so the endpoint answers `200 OK` with
  `{"last_coefficients":[null,null,null], ...}` — `learning_rate=0.1` on the sample
  data set above is already enough to trigger it. The range check on `learning_rate`
  only rejects values that are not positive and finite; it cannot know what is too
  large for a given data set. Compare `J_after_learning` against `J_before_learning`
  to catch this until it is handled server-side.
- **Request bodies are capped at 32 MB** by the `DefaultBodyLimit` layer in
  `main.rs`, and going over does not surface as `413 Payload Too Large`. The limit
  makes the body stream fail part-way, and the multipart extractor reports that as a
  plain `400` with `Error parsing multipart/form-data request` — the very message a
  genuinely malformed body produces, so a client cannot tell the two apart. Reading
  the field with a size check of its own would be needed to answer `413` properly.
- **Training blocks an async worker thread.** The descent is synchronous CPU work
  running directly in the handler; it belongs in `tokio::task::spawn_blocking`.
- **The gradient loop recomputes predictions per parameter.** `dJ/da_j` calls the
  hypothesis for every sample, and it is called once per feature, making an
  iteration `O(m * n^2)` where `O(m * n)` would do. Caching the per-sample errors
  once per iteration fixes it.
- **No feature scaling.** By design — see the top of this file.
