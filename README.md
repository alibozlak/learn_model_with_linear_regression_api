# learn_model_with_linear_regression_api

Multivariable linear regression trained with batch gradient descent, exposed over
HTTP. Written from scratch — no `ndarray`, no `linfa`, no autodiff — so that every
step of the math is visible in the source.

The model itself still does no scaling — "without feature scaling" in the module
name is literal, and the long narrow valley that unscaled features carve out is
worth seeing. The scaling now happens one step earlier instead: `/train` hands the
data set to `data_manipulate_api` first and trains on what comes back, which is
what lets the learning rate be a sane number. None of that reaches the caller: the
coefficients are lifted back into the payload's own units before the reply leaves.
See [The training chain](#the-training-chain).

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
to `data_manipulate_api`, which divides every column by the power of ten that makes
its first value single-digit, and the descent runs on that:

```text
Angular  ──POST /train──>  learn_model  ──POST /manipulate-datas──>  data_manipulate
                                │                                          │
                                │  <────── scaled columns + ratios ────────┘
                                │
                                ├──> descent on scaled data ──> coefficients (scaled)
                                │
                                └──> lifted back with ratios ──> coefficients (real)
```

The point is the learning rate. On the sample data set below, raw magnitudes need
`0.000003`, and a step even one order larger diverges. On the rescaled columns the
ceiling sits between `0.007` and `0.008` — three orders of magnitude more room — and
20000 iterations reach a lower cost than a million did before.

Coefficients come back **in the units the data set was written in**. The descent
still runs in the scaled space, but the handler lifts its result out before
answering, with the `ratios` the scaler reported alongside the scaled columns:

```text
a_j = a'_j * 10^(r_y - r_j)      b = b' * 10^r_y
```

Both costs are measured the same way, against the unscaled samples, on a copy of the
model that never trains — so `J_before_learning` and `J_after_learning` are readable
against the magnitudes in the payload and not only against each other. `ratios` is
not reported any more, because there is nothing left for the caller to do with it.

There is no starting point to send. The descent always begins at the origin, because
a caller-supplied one would be expressed in the units of the payload, which is not
the space the descent runs in once the scaler has been through it.

> **Caveat on the exponent.** The scaler reads a column's exponent off that column's
> *first* value and then divides the whole column by it. The conversion above is
> therefore exact — every row of a column shares one exponent — but a column whose
> first value is small next to the rest is barely scaled at all and goes on driving
> the curvature. See [Known limitations](#known-limitations).

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
docker run --rm -p 127.0.0.1:3000:3000 \
  -e DATA_MANIPULATE_URL=http://<scaler-host>:3001/manipulate-datas \
  linear-regression-api
```

The image is always the optimised build, so the `--release` caveat above does not
apply to it. Released builds are pushed to Docker Hub under `bozlak`; a deployment
pulls the tag from there rather than building from this tree.

**Composing the stack is not this repository's job.** The file that runs this image
alongside the scaler and a reverse proxy lives in its own directory, outside any of
the three projects, and pulls all three from the Hub by tag. What it has to get right
is described under [Keeping the pair private](#keeping-the-pair-private).

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
| `DATA_MANIPULATE_URL` | `http://127.0.0.1:3001/manipulate-datas` | The scaler's endpoint. A compose stack points this at the scaler's service name. |
| `ALLOWED_ORIGINS` | `http://localhost:4200` | Comma-separated origins the browser may read a reply from — see [CORS](#cors). |

## Keeping the pair private

Only the Angular app is meant to reach these services, and what enforces that is the
network, not a header. The compose file that arranges this lives outside these
repositories; what it has to get right is the following.

**Neither Rust service should publish a port.** With no `ports:` entry the only thing
that can reach them is another service on the same private network. **A reverse proxy
is the single published thing**, pinned to `127.0.0.1` on the host side, so nothing off
the machine reaches any of it. Routing `/api/linear-regression/` to this service with
the prefix stripped makes `/api/linear-regression/train` arrive as `/train`.

Two settings on that proxy matter, because their defaults break this endpoint:

| Setting | Default | Why it has to change |
| --- | --- | --- |
| `client_max_body_size` | `1m` in nginx | A data set is rejected with the proxy's own `413` long before this service's 32 MB limit applies |
| `proxy_read_timeout` | `60s` in nginx | A large `loop_count` holds the connection open for the whole descent and is cut off with a `504` |

**Both services must keep listening on `0.0.0.0` inside their containers.** That is not
a hole, and loopback there is not an improvement — it is unreachable from the host and
from every other container alike, so it would break the chain and buy nothing. Verified
by running the release binary in a `debian:bookworm-slim` container:

| Inside the container | Host mapping | Result |
| --- | --- | --- |
| `BIND_ADDR=127.0.0.1:3001` | `-p 13001:3001` | connection refused |
| `BIND_ADDR=0.0.0.0:3001` | `-p 127.0.0.1:13001:3001` | `405` — service answers |

The second row is already private to the machine: the `127.0.0.1:` on the **host** side
of the mapping is what does that, not the bind address inside the container.

Measured on a stack arranged this way:

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

Serving the Angular build from that same proxy puts the page and the API on one origin,
which drops the preflight and makes the CORS headers moot. `ng serve` on `:4200`
remains a separate origin and still needs them.

What none of this does is prove the caller is the Angular app. An SPA ships its own
source to the browser, so any key it could present is readable by whoever opens
devtools; `curl` from this machine reaches the proxy exactly as the page does.
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

| Parameter | Default | Accepted         | Meaning |
| --- | --- |------------------| --- |
| `learning_rate` | `0.003` | finite and `> 0` | Step size of each gradient descent update |
| `loop_count` | `20000` | `1` to `1000000` | How many iterations to run |

The defaults are sized for the rescaled columns the descent runs on, not the raw
payload. On the sample data set the ceiling before divergence sits between `0.007`
and `0.008`, and the default stays under it with room to spare, because the ceiling
is data-dependent: a column the scaler barely moves — one already below 10, or one
whose first value is small next to the rest — still drives the curvature. Anything
above the default is expected to be data-dependent in exactly this way. Measured on
that data set, against a live `data_manipulate_api`:

| `learning_rate` | `loop_count` | `J_after_learning` |
| --- | --- | --- |
| `0.008` | 20000 | `0` — diverged, `null` coefficients |
| `0.007` | 20000 | `883121` |
| `0.005` | 20000 | `883121` |
| `0.003` (default) | 20000 | `883218` |
| `0.003` (default) | 1000 | `2273972` |
| `0.000003` | 1000000 | `2274468` |

The costs are in the caller's own units now, so they are large where the data is
large: `J_before_learning` on this set is `1818750000`, and the converged fit sits
three orders of magnitude below it. That last row is what this endpoint needed before
the scaling hop existed — the default now reaches a cost two and a half times lower
with a thousand times the step and a fiftieth of the iterations.

**Response** — `200 OK`, `Content-Type: application/json`, single-line:

```json
{"J_before_learning":1818750000,"J_after_learning":883218,"last_coefficients":[245.86312190187883,2566.50059820144,7644.724306977841]}
```

The handler returns `Json<TrainingResult>` rather than a `String`, which is what sets
that content type; error replies stay `text/plain`, as the table below describes.

| Field | Meaning |
| --- | --- |
| `J_before_learning` | Cost at the origin, where every run starts. A whole number — see below. |
| `J_after_learning` | Cost the run ended on. Comparing the two is the quickest way to tell whether the learning rate was sane, with one exception: a diverged run reports `0`. |
| `last_coefficients` | The fit, as `[a_1, ..., a_n, b]` with the bias last, so `n + 1` long. |

Everything here is in the units the data set was written in. The descent runs on the
scaler's columns, but the handler lifts the coefficients back out with `ratios` and
measures both costs against the unscaled samples, so there is nothing left to
convert and nothing about the scaling hop shows up in the reply. The fit above reads
as roughly `246` per square metre, `2567` per room and a base of `7645`.

The two costs are `u128`, so they arrive as whole numbers: the mean squared error is
summed in `f64` and truncated once, at the end. At the magnitudes a real data set
produces that loses nothing worth having, but a cost below 1 reads as `0` — see
[Known limitations](#known-limitations).

`ratios` was part of this reply until the conversion moved server-side; a client that
still reads it will find it gone, along with `scaled_last_coefficients`, which is now
`last_coefficients` and no longer scaled.

**Errors** — `4xx`/`5xx` with a plain-text body describing the problem:

| Status | Body | Cause |
| --- | --- | --- |
| `400` | ``Invalid `boundary` for `multipart/form-data` request`` | The body was not sent as multipart — e.g. raw JSON via `--data-binary` |
| `400` | `Form hasn't a 'dataset' named field !!` | Multipart body carries no field called `dataset` |
| `400` | `Failed to deserialize query string: ...` | A query parameter is not of the expected type, e.g. `loop_count=abc` |
| `400` | `learning_rate must be a finite number greater than 0, got X` | `learning_rate` is zero, negative, `NaN` or infinite |
| `400` | `loop_count must be at least 1` | `loop_count` is `0` |
| `400` | `loop_count must not exceed 1000000, got X` | `loop_count` is above the cap |
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
curl -X POST -F "dataset=@dataset.json" "http://localhost:3000/train?learning_rate=0.005&loop_count=5000"
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

The layering is one-directional: the model knows nothing about JSON, and
`json_converter` knows nothing about HTTP. `json_converter` validates the payload
before the model's constructor is reached, so data arriving from outside can never
trigger the panics that constructor documents. The coefficient vector is no longer
one of the things that could: the handler builds it as `vec![0.0; n + 1]` from the
feature count the scaler reported, which is checked against the length of `ratios`
first, so it cannot reach the constructor at the wrong length.

The handler is the only place that sees both spaces at once. It builds three models
around a single descent — one on the unscaled data at the origin for the cost
before, one on the scaled data that actually trains, one on the unscaled data
holding the converted coefficients for the cost after — which is why the model's
`coefficients` field is public: it is how the starting point is carried from the
first to the second.

## Known limitations

These are either deliberate or on the roadmap — listed so nobody has to rediscover
them.

- **A column's exponent is read off its first row.** `data_manipulate_api` takes the
  power of ten from each column's first value and divides the whole column by it, so
  the mapping back is exact — but how well a column is conditioned depends on which
  row happens to be first. The sample data set's area column
  `[55, 72, 90, 110, 130, 165]` comes back as `[5.5, ..., 16.5]`, still spanning a
  factor of three and still over 10 at the top, and that is what holds the
  learning-rate ceiling below `0.008`. A column whose first value is small next to
  the rest is scaled least where it would help most.
- **No divergence guard.** If the learning rate is too large for the data's scale,
  the coefficients overflow to `NaN`. `serde_json` writes non-finite floats as
  `null`, so the endpoint answers `200 OK` with
  `{"last_coefficients":[null,null,null], ...}` — `learning_rate=0.008` on the sample
  data set is already enough, even after the scaling hop. The range check on
  `learning_rate` only rejects values that are not positive and finite; it cannot
  know what is too large for a given data set. The costs will not give it away
  either: `J_after_learning` is a `u128` and `NaN as u128` is `0`, so a run that
  blew up reports the lowest cost there is, right next to a `J_before_learning` of
  nine figures. The `null` coefficients are the signal to check until this is
  handled server-side.
- **The costs are whole numbers.** `J` returns `u128`, so the mean squared error
  arrives truncated and anything below 1 reads as `0`. It is summed in `f64` and cast
  once at the end, which is what keeps the sign of each residual intact — casting
  them one by one would saturate every negative one to zero and quietly drop the
  samples the model under-estimates. On a data set whose outputs are small, though,
  a `0` here means "under one", not "exact".
- **Scaling only moves by powers of ten.** A column already below 10 keeps an
  exponent of 0 and is passed through untouched — the room count on the sample data
  set, spanning 1 to 5, is one such column. The hop cannot flatten a valley it is not
  allowed to rescale, so what is left of the valley after scaling is what sets the
  ceiling on `learning_rate`. On this data set that is the area column rather than
  the room count: at `[5.5, ..., 16.5]` it still carries an order of magnitude more
  curvature than the rooms do.
- **The chain costs a round trip and three copies.** The whole payload crosses to
  `data_manipulate_api` and the scaled version comes back, and the handler then
  parses the raw JSON a second time locally, because the costs are measured against
  the caller's own numbers. A large data set is therefore held as the raw text, the
  scaled vectors and the unscaled vectors at once, and travels the loopback interface
  once each way. The 32 MB body limit applies to what arrives here; nothing caps what
  the scaler returns.
- **The payload's validation effectively happens in the other service.**
  `json_converter` still runs here, but after the hop, and its result is `unwrap()`ed.
  Everything it would reject — an empty set, a sample-count mismatch, a ragged row —
  `data_manipulate_api` has already answered `400` for, and that status is passed
  straight through, so the `unwrap()` is unreachable in practice. What keeps it
  unreachable is the two services agreeing on those checks, and nothing in this
  repository enforces the agreement. Parsing before the hop would make this service's
  own error the one the caller sees and would save a round trip on data that was
  never usable.
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
