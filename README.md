# learn_model_with_linear_regression_api

Multivariable linear regression trained with batch gradient descent, exposed over
HTTP. Written from scratch — no `ndarray`, no `linfa`, no autodiff — so that every
step of the math is visible in the source.

The model itself still does no scaling — "without feature scaling" in the module
name is literal, and the long narrow valley that unscaled features carve out is
worth seeing. The scaling now happens one step earlier instead: `/train` hands the
data set to `data_manipulate_api` first and trains on what comes back, which is
what lets the learning rate be a sane number. See
[The training chain](#the-training-chain).

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

## The training chain

`POST /train` does not train on the data set it is given. It forwards the payload
to `data_manipulate_api`, which divides each column by a power of ten until it is
single-digit, and the descent runs on that:

```text
Angular  ──POST /train──>  learn_model  ──POST /manipulate-datas──>  data_manipulate
                                │                                          │
                                │  <────── scaled columns + ratios ────────┘
                                │
                                └──> descent on scaled data ──> coefficients (scaled)
```

The point is the learning rate. On the sample data set below, raw magnitudes need
`0.000003`, and a step even one order larger diverges. On the rescaled columns
`0.05` is still stable and `0.01` is comfortable — roughly four orders of magnitude
more room — and 5000 iterations reach a lower cost than a million did before.

Coefficients come back **in the scaled space**, as `scaled_last_coefficients`, with
`ratios` alongside them. Lifting them into the units the data set was written in is
the caller's to do:

```text
a_j = a'_j * 10^(r_y - r_j)      b = b' * 10^r_y
```

There is no starting point to send. The descent always begins at the origin, because
a caller-supplied one would be expressed in the units of the payload, which is not
the space the descent runs in once the scaler has been through it.

> **Caveat on the mapping.** The scaler picks its exponent per value, not per column,
> so `ratios` describes only the first sample. On a column of mixed magnitudes the
> conversion above is wrong for every row whose exponent differs from row 0's — see
> [Known limitations](#known-limitations).

The hop is the only way in for the scaler: nothing else needs to reach it, which is
what [Keeping the pair private](#keeping-the-pair-private) relies on.

## Requirements

- Rust (edition 2024)
- No system dependencies
- A reachable `data_manipulate_api` — `/train` answers `502` without it

Docker alone is enough if you would rather not install a toolchain — see
[Docker](#docker) below.

## Running

```bash
cargo run
```

The server binds `127.0.0.1:3000` and prints the address it is listening on. Set
`BIND_ADDR` to change it — see [Configuration](#configuration). Use
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
- **The listening address comes from `BIND_ADDR`**, which the image sets to
  `0.0.0.0:3000`; the binary's own default is loopback, and a container bound to
  loopback is unreachable even from its own host. Remapping from the host is still
  the simpler way to move the port: `-p 127.0.0.1:8080:3000`. Keep the `127.0.0.1:`
  prefix — see [Keeping the pair private](#keeping-the-pair-private).

`.dockerignore` keeps `target/` out of the build context. It grows to a few hundred
megabytes as soon as the project is built locally, and every byte would otherwise be
handed to the daemon on each build.

## Configuration

All optional, all read once at startup:

| Variable | Default | Meaning |
| --- | --- | --- |
| `BIND_ADDR` | `127.0.0.1:3000` | Where the server listens. Loopback keeps it off the network; a container has to override it to `0.0.0.0:3000` or nothing reaches it, even from its own host. |
| `DATA_MANIPULATE_URL` | `http://127.0.0.1:3001/manipulate-datas` | The scaler's endpoint. Compose points this at the service name. |
| `ALLOWED_ORIGINS` | `http://localhost:4200` | Comma-separated origins the browser may read a reply from — see [CORS](#cors). |

## Keeping the pair private

Only the Angular app is meant to reach these services, and what enforces that is the
network, not a header.

**Neither Rust service publishes a port.** Under `docker-compose.yml` they carry no
`ports:` entry at all, so the only thing that can reach them is another service on the
same private network. **`gateway`, an nginx, is the single published thing**, and it is
published to `127.0.0.1:8080` — so nothing off this machine reaches any of it. It
proxies `/api/linear-regression/` to `learn-model`, stripping the prefix, so
`/api/linear-regression/train` arrives as `/train`.

Both services still listen on `0.0.0.0` **inside** their containers, and that is not a
hole. A container bound to loopback is unreachable from its own host and from every
other container; publishing a port does not change it. Running the release binary in a
`debian:bookworm-slim` container:

| Inside the container | Host mapping | Result |
| --- | --- | --- |
| `BIND_ADDR=127.0.0.1:3001` | `-p 13001:3001` | connection refused |
| `BIND_ADDR=0.0.0.0:3001` | `-p 127.0.0.1:13001:3001` | `405` — service answers |

Binding loopback inside the containers would simply break the chain, and buy nothing:
the second row is already private to the machine, because of the `127.0.0.1:` on the
**host** side of the mapping.

```bash
docker build -t data_manipulate_api ../data_manipulate_api
```

```bash
docker compose up --build
```

Measured on the stack that comes up:

| Request | Result |
| --- | --- |
| `POST 127.0.0.1:8080/api/linear-regression/train` | `200`, trained |
| `127.0.0.1:3000` and `127.0.0.1:3001` from the host | connection refused |
| `<this machine's LAN address>:8080` | connection refused |

`docker compose ps` should show one published port and no others:

```text
gateway           127.0.0.1:8080->80/tcp
learn-model       3000/tcp
data-manipulate   3001/tcp
```

Mounting an Angular build into the gateway — the commented-out volume in
`docker-compose.yml` — puts the page and the API on one origin, which drops the
preflight and makes the CORS headers moot. `ng serve` on `:4200` remains a separate
origin and still needs them.

What none of this does is prove the caller is the Angular app. An SPA ships its own
source to the browser, so any key it could present is readable by whoever opens
devtools; `curl` from this machine reaches the gateway exactly as the page does.
Restricting by network is real, and it is the honest limit of what is enforceable
here — anything stronger means authenticating the user, not the app.

## CORS

A browser will not hand a cross-origin response to JavaScript unless the reply carries
`Access-Control-Allow-Origin`, so a page served from anywhere other than this server's
own origin cannot read an answer without the `CorsLayer` in `main.rs`.

The layer sits outermost, which is what lets a preflight be answered before the body
limit or the router see the request. Without it a browser's `OPTIONS /train` reaches a
router that carries only a `POST` route, comes back `405`, and the preflight fails
before the real request is ever sent.

Allowed origins come from `ALLOWED_ORIGINS`, comma-separated. The default is
`http://localhost:4200`, which is where `ng serve` puts an Angular dev server:

```bash
ALLOWED_ORIGINS="http://localhost:4200,https://bozlak.dev" cargo run --release
```

```bash
docker run --rm -p 3000:3000 -e ALLOWED_ORIGINS="https://bozlak.dev" linear-regression-api
```

Only `POST` is advertised and only `Content-Type` is accepted as a request header. A
`multipart/form-data` upload is a content type the CORS spec safelists, so a browser
sends it without a preflight at all — the preflight path matters for clients that send
something else, such as a raw `application/json` body.

`curl` neither sends a preflight nor looks at any of these headers, so an endpoint that
answers from the terminal can still be unreachable from a browser. That gap is the
usual reason a client reports a status of 0 while the server log shows nothing wrong.

**This is not access control.** The check happens in the browser, against the response.
A request from a disallowed origin — or from anything that is not a browser — still
reaches the handler and still trains the model; the browser merely refuses to let the
page read the answer. An authenticating proxy in front is what restricts the endpoint,
rather than making it usable from one page.

## API

### `POST /train`

Trains a model on the data set in the request and returns the coefficients it
converged to. The server keeps no state between requests, and the descent always
starts at the origin.

**Request** — `multipart/form-data` carrying a single field named `dataset`, whose
content is the training set as JSON. `inputs[i]` is one sample's feature values and
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

There is no `initial_coefficients` field. Earlier versions required one; a payload
that still carries it is accepted and the field ignored, so existing data sets do not
have to be edited.

**Query parameters** — both optional, both checked before any training starts:

| Parameter | Default | Accepted | Meaning |
| --- | --- | --- | --- |
| `learning_rate` | `0.01` | finite and `> 0` | Step size of each gradient descent update |
| `loop_count` | `20000` | `1` to `5000000` | How many iterations to run |

The defaults are sized for the rescaled columns the descent runs on, not the raw
payload. On the sample data set the ceiling before divergence sits near `0.05`, and
the defaults stay under it with room to spare, because the ceiling is
data-dependent: a column the scaler leaves alone, being already below 10, still
drives the curvature. Measured on that data set:

| `learning_rate` | `loop_count` | `J_after_learning` |
| --- | --- | --- |
| `0.08` | 20000 | diverges — `null` coefficients |
| `0.05` | 5000 | `0.0090` |
| `0.01` | 20000 | `0.0093` |
| `0.000003` | 1000000 | `0.0500` |

That last row is what this endpoint needed before the scaling hop existed.

**Response** — `200 OK`, `Content-Type: application/json`, single-line:

```json
{"J_before_learning":18.187500000000004,"J_after_learning":0.01820230981958408,"ratios":[1,0,4],"scaled_last_coefficients":[-0.05897920763610215,0.8597882776149124,1.7570875999268627]}
```

The handler returns `Json<TrainingResult>` rather than a `String`, which is what sets
that content type; error replies stay `text/plain`, as the table below describes.

| Field | Meaning |
| --- | --- |
| `J_before_learning` | Cost at the origin, where every run now starts. |
| `J_after_learning` | Cost the run ended on. Comparing the two is the quickest way to tell whether the learning rate was sane. |
| `ratios` | The exponent each column was divided by, one per feature with `outputs` last. |
| `scaled_last_coefficients` | The fit, as `[a_1, ..., a_n, b]` with the bias last, so `n + 1` long. |

Everything here is in the rescaled space the descent minimises over, so the numbers
are comparable to each other but not to the magnitudes in the payload. Converting is
the caller's job: `10^(r_y - r_j)` for a feature's coefficient, `10^r_y` for the bias,
`10^(2 * r_y)` for a cost, with `r_y` the last entry of `ratios`.

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
| `400` | `"inputs" is empty, there is nothing to train on` | `inputs` is `[]` |
| `400` | `sample count mismatch: N input sample(s) but M output(s)` | `inputs` and `outputs` differ in length |
| `400` | `sample K has F feature(s) while the first sample has E` | Rows of differing width |
| `400` | ``Error parsing `multipart/form-data` request`` | The multipart body is malformed, or the request went over the 32 MB body limit |
| `4xx` | `the scaling service rejected the data set: ...` | `data_manipulate_api` refused the payload; its status and message are passed through, since the data it refused is the caller's |
| `502` | `could not reach the scaling service at URL: ...` | `data_manipulate_api` is down or the URL is wrong — see [Configuration](#configuration) |
| `502` | `the scaling service answered STATUS: ...` | It replied, but with a `5xx` |
| `502` | `could not parse the scaling service's reply: ...` | Its reply was not the expected JSON |
| `502` | `the scaling service returned N ratios for M feature(s), expected ...` | The scaled data does not match the payload's shape |
| `500` | serialisation error | The result could not be encoded |

## Usage with curl

Send a data set held in a file. `-F` is what makes curl build the multipart body and
set the `Content-Type` boundary along with it; the `@` tells it to read a file rather
than take the text literally:

```bash
curl -X POST -F "dataset=@dataset.json" http://localhost:3000/train
```

Inline, for a quick check — no `@` this time, so the JSON is used as the value:

```bash
curl -X POST -F 'dataset={"inputs":[[55.0,1.0],[130.0,4.0]],"outputs":[23000.0,48500.0]}' http://localhost:3000/train
```

Override either hyperparameter from the query string. Quote the URL, otherwise the
shell swallows everything from the `&` onwards:

```bash
curl -X POST -F "dataset=@dataset.json" "http://localhost:3000/train?learning_rate=0.05&loop_count=5000"
```

Pretty-print the response by piping through `jq`:

```bash
curl -s -X POST -F "dataset=@dataset.json" http://localhost:3000/train | jq
```

## Layout

```text
src/
├── main.rs                               Router, body limit, CORS, server bootstrap
├── train_endpoint.rs                     The POST /train handler
├── train_params.rs                       Query parameters and their validation
├── data_manipulate_client.rs             The hop out to data_manipulate_api
├── json_converter.rs                     JSON <-> the model's vectors, plus validation
└── learning_without_feature_scaling.rs   The model: hypothesis, cost, gradients, descent
```

`docker-compose.yml` and `nginx.conf` sit at the root: the compose file wires the two
services onto a private network, and `nginx.conf` is the only thing the outside talks
to. See [Keeping the pair private](#keeping-the-pair-private).

The layering is one-directional: the model knows nothing about JSON, and
`json_converter` knows nothing about HTTP. `json_converter` validates the payload
before the model's constructor is reached, so data arriving from outside can never
trigger the panics that constructor documents. The coefficient vector is no longer
one of the things that could: the handler builds it as `vec![0.0; n + 1]` from a
feature count validation has already pinned, so it cannot reach the constructor at
the wrong length.

## Known limitations

These are either deliberate or on the roadmap — listed so nobody has to rediscover
them.

- **`ratios` does not describe a column, so the mapping back is wrong.** The scaler
  picks its exponent per value rather than per column, and reports only the first
  sample's. A column of mixed magnitudes is therefore divided unevenly while `ratios`
  claims a single exponent for it. On the sample data set the area column
  `[55, 72, 90, 110, 130, 165]` comes back as `[5.5, 7.2, 9.0, 1.1, 1.3, 1.65]` — the
  order reversed between rows 2 and 3 — and the descent fits that, ending on a
  *negative* coefficient for area. Converting with `ratios` then predicts rows 0-2 to
  within a percent and rows 3-5 out by 9-17%. Any column whose values all share a
  digit count is unaffected; fixing the rest means scaling per column, which
  `data_manipulate_api` deliberately does not do.
- **No divergence guard.** If the learning rate is too large for the data's scale,
  the coefficients overflow to `NaN`. `serde_json` writes non-finite floats as
  `null`, so the endpoint answers `200 OK` with
  `{"scaled_last_coefficients":[null,null,null], ...}` — `learning_rate=0.08` on the sample
  data set is already enough, even after the scaling hop. The range check on
  `learning_rate` only rejects values that are not positive and finite; it cannot
  know what is too large for a given data set. Compare `J_after_learning` against
  `J_before_learning` to catch this until it is handled server-side.
- **Scaling only moves by powers of ten.** A column already below 10 keeps an
  exponent of 0 and is passed through untouched, so it goes on driving the curvature
  and setting the ceiling on `learning_rate`. On the sample data set the room count,
  spanning 1 to 5, is what holds the ceiling near `0.05` — the hop cannot flatten a
  valley it is not allowed to rescale.
- **The chain costs a round trip and a second copy.** The whole payload crosses to
  `data_manipulate_api` and the scaled version comes back, so a large data set is
  held in memory twice over and travels the loopback interface once each way. The
  32 MB body limit applies to what arrives here; nothing caps what the scaler
  returns.
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
