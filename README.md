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
                                └──> descent on scaled data ──> coefficients mapped back
```

The point is the learning rate. On the sample data set below, raw magnitudes need
`0.000003`, and a step even one order larger diverges. On the rescaled columns
`0.05` is still stable and `0.01` is comfortable — roughly four orders of magnitude
more room — and 5000 iterations reach a lower cost than a million did before.

Coefficients come back in **the caller's units**, not the scaled ones. `ratios` is
what makes that exact: with one exponent per column the rescaling is linear, so

```text
a_j = a'_j * 10^(r_y - r_j)      b = b' * 10^r_y
```

recovers the original-unit fit, and the same conversion runs in reverse on
`initial_coefficients` on the way in. That reverse step is what keeps a run
resumable — a split run still lands on the same cost as the undivided one, to the
last bit.

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

Only the Angular app is meant to reach these two services, and the arrangement that
actually enforces that is a network one, not a header one.

**`data_manipulate_api` is never exposed.** Nothing but this service calls it, so it
publishes no port: running natively it binds `127.0.0.1:3001`, and under Compose it
sits on an internal network with no `ports:` entry at all. There is no address for a
browser or another machine to reach it on.

**`learn_model` is the single entry point**, published to `127.0.0.1:3000` rather
than `0.0.0.0:3000`. The browser on this machine reaches it; nothing off the machine
does. `docker-compose.yml` pins the host side of the mapping for exactly this
reason — dropping the `127.0.0.1:` prefix would publish on every interface and undo
it.

```bash
docker build -t data_manipulate_api ../data_manipulate_api
```

```bash
docker compose up --build
```

Verify the isolation holds, substituting this machine's own LAN address:

```bash
curl -m 3 http://$(hostname -I | awk '{print $1}'):3000/train
```

That has to fail to connect. So does the same call against port 3001, and 3001 has
to fail from `127.0.0.1` too when running under Compose.

What this does **not** do is prove the caller is the Angular app. An SPA ships its
own source to the browser, so any key it could present is readable by whoever opens
devtools; `curl` from this machine reaches `learn_model` exactly as the page does.
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
{"last_coefficients":[216.01962911378516,3391.8114446308246,8266.719117891573],"J_before_learning":18.187500000000004,"J_after_learning":0.009272647375726672,"ratios":[2,0,4],"scaled_last_coefficients":[2.1601962911378516,0.33918114446308245,0.8266719117891572]}
```

The handler returns `Json<TrainingResult>` rather than a `String`, which is what sets
that content type; error replies stay `text/plain`, as the table below describes.

| Field | Units | Meaning |
| --- | --- | --- |
| `last_coefficients` | caller's | The fit, as `[a_1, ..., a_n, b]` with the bias last. Accepted verbatim as the next request's `initial_coefficients`. |
| `J_before_learning` | scaled | Cost of the coefficients that were *sent in*, so it is the cost at the origin only when zeros were sent. |
| `J_after_learning` | scaled | Cost the run ended on. Comparing the two is the quickest way to tell whether the learning rate was sane. |
| `ratios` | — | The exponent each column was divided by, one per feature with `outputs` last, so a caller can redo the mapping themselves. |
| `scaled_last_coefficients` | scaled | The coefficients as the descent left them, before the mapping back. Useful for telling a scaling problem apart from a training one. |

Both costs are measured in the rescaled space the descent minimises over, so they are
comparable to each other but not to the magnitudes in the payload. `10^(2 * r_y)`
converts one to the caller's units, `r_y` being the last entry of `ratios`.

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

Inline, for a quick check — no `@` this time, so the JSON is used as the value.
`initial_coefficients` has to be there even when it is all zeros:

```bash
curl -X POST -F 'dataset={"inputs":[[55.0,1.0],[130.0,4.0]],"outputs":[23000.0,48500.0],"initial_coefficients":[0.0,0.0,0.0]}' http://localhost:3000/train
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

### Resuming a run

Because `last_coefficients` comes back in the layout `initial_coefficients` expects,
`jq` can splice one run's answer into the next request's payload:

```bash
curl -s -X POST -F "dataset=@dataset.json" "http://localhost:3000/train?loop_count=5000" | jq --slurpfile d dataset.json '$d[0] + {initial_coefficients: .last_coefficients}' > next.json
```

```bash
curl -s -X POST -F "dataset=@next.json" "http://localhost:3000/train?loop_count=5000"
```

Two runs of 5000 iterations on the sample data set land on exactly what a single run
of 10000 reaches — the same coefficients to the last bit, since nothing but the
starting point distinguishes the second call from a continuation of the first:

```text
0     -> 5000   {"last_coefficients":[122.62288240414479,5978.939928210685,10196.804398361628],"J_before_learning":18.187500000000004,"J_after_learning":0.01688775898211305}
5000  -> 10000  {"last_coefficients":[169.41234066449698,4682.85206326374,9229.881316566152],"J_before_learning":0.016887758982113074,"J_after_learning":0.011891211803789228}

0     -> 10000  {"last_coefficients":[169.41234066449698,4682.85206326374,9229.881316566152],"J_before_learning":18.187500000000004,"J_after_learning":0.011891211803789228}
```

This survives the scaling hop because the answer is mapped back into the caller's
units and mapped forward again on the way in, and the two conversions are exact
inverses. The coefficients match to the last bit; the resumed leg's
`J_before_learning` sits an ulp or two off the previous leg's `J_after_learning`,
which is the float round trip through JSON, not drift in the descent.

The one thing this buys that a longer `loop_count` does not is a look at the cost
part-way: each leg reports its own `J_before_learning` and `J_after_learning`, so a
run that is drifting towards divergence shows it before the whole budget is spent.

## Layout

```text
src/
├── main.rs                               Router, body limit, CORS, server bootstrap
├── train_endpoint.rs                     The POST /train handler
├── train_params.rs                       Query parameters and their validation
├── data_manipulate_client.rs             The hop out to data_manipulate_api
├── unit_mapping.rs                       Coefficients between caller's and scaled units
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
  `{"last_coefficients":[null,null,null], ...}` — `learning_rate=0.08` on the sample
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
