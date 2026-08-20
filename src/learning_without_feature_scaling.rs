/// Multivariable linear regression trained with batch gradient descent,
/// deliberately *without* feature scaling.
///
/// The model function (hypothesis) is
///
/// ```text
/// f(x) = a_1 * x_1 + a_2 * x_2 + ... + a_n * x_n + b
/// ```
///
/// and the cost being minimised is the mean squared error
///
/// ```text
/// J = (1/m) * sum over i of ( f(x^(i)) - y^(i) )^2
/// ```
///
/// Since the features keep their original ranges (square meters in the tens or
/// hundreds versus a room count in single digits), the cost surface is a long
/// and narrow valley: gradient descent only stays stable with a very small
/// learning rate and needs many iterations. Nothing here does anything about
/// that: a caller is free to hand the constructor columns that have already
/// been divided down — which is what `train_endpoint` does — but this model
/// cannot tell such a data set apart from a raw one, and reports every number
/// in whichever space it was given.
pub struct WithoutFeatureScaling {

    /// Training inputs: `m` samples of `n` features each, so `real_inputs[i][j]`
    /// is feature `j` of sample `i` (`x_j^(i)`).
    real_inputs : Vec<Vec<f64>>,

    /// The expected output (`y^(i)`) of every sample, in the same order as
    /// [`Self::real_inputs`].
    real_outputs : Vec<f64>,

    /// Sample count.
    m : usize,

    /// Feature count of a single sample.
    n : usize,

    /// The parameters being learned, laid out as `[a_1, ..., a_n, b]`. The last
    /// slot (index `n`) is the bias `b`, which is why the length is `n + 1`.
    ///
    /// Public so that a starting point can be carried from one model to the
    /// next: `train_endpoint` reads the cost at the origin off a model built on
    /// the unscaled data and then hands these same coefficients to the model
    /// that runs the descent on the scaled data. It is the one field that
    /// escapes [`Self::validate_data_set`] — assigning a vector of any length
    /// other than `n + 1` leaves the struct in the state that check exists to
    /// reject, and the next prediction indexes out of bounds.
    pub coefficients : Vec<f64>
}

impl WithoutFeatureScaling {

    /// Builds a model from a training set and a starting point for the
    /// coefficients.
    ///
    /// `f_initial_coefficients` is the `[a_1, ..., a_n, b]` vector the descent
    /// starts from; it may be all zeros, or the output of an earlier training
    /// run so that training can be continued where it left off.
    ///
    /// Validation also derives `m` and `n` from the data itself, so the struct
    /// can never hold sizes that disagree with the vectors it stores.
    ///
    /// Panics if the data set is inconsistent — see [`Self::validate_data_set`].
    pub fn new(
        real_inputs : Vec<Vec<f64>>,
        real_outputs : Vec<f64>,
        f_initial_coefficients : Vec<f64>
    ) -> WithoutFeatureScaling {
        let (real_inputs, real_outputs, m, n, initial_coefficients)
            = Self::validate_data_set(real_inputs, real_outputs, f_initial_coefficients);
        Self { real_inputs, real_outputs, m, n, coefficients: initial_coefficients }
    }

    /// Runs batch gradient descent for `loop_count` iterations and returns the
    /// coefficients reached at the end.
    ///
    /// Reading the cost is the caller's: [`Self::J`] answers it at any point,
    /// and the two readings a run is judged by are not necessarily taken here
    /// at all — `train_endpoint` takes both of them on the unscaled data set,
    /// against a copy of the model that never trains.
    ///
    /// Each iteration updates every parameter once:
    ///
    /// ```text
    /// a_j := a_j - learning_rate * dJ/da_j
    /// b   := b   - learning_rate * dJ/db
    /// ```
    ///
    /// "Batch" means every derivative is computed over all `m` samples, so one
    /// iteration is one pass over the whole training set. The loop runs a fixed
    /// number of times; there is no convergence check that could stop it early.
    ///
    /// The new values are collected in `temp_coefficients` instead of being
    /// written straight into `self.coefficients`, which keeps the update
    /// *simultaneous*: every derivative of one iteration is computed from the
    /// same, still unchanged coefficients. Writing in place would make the
    /// derivative of `a_2` already see the updated `a_1`, which is a different
    /// algorithm. The step only takes effect once the whole iteration is done.
    ///
    /// The two vectors then trade places rather than one being copied over the
    /// other. Every slot of `temp_coefficients` is written before it is read
    /// again, so what the swap leaves behind — the iteration's old
    /// coefficients — is scratch space either way, and the step costs two
    /// pointer moves instead of `n + 1` float copies.
    pub fn train_model(&mut self, learning_rate : f64, loop_count : usize) -> Vec<f64>
    {
        let mut temp_coefficients  = vec![0.0; self.n + 1];
        for _ in 0..loop_count {
            for j in 0..self.n {
                temp_coefficients[j] = self.coefficients[j] - learning_rate * self.dJ_daj(j);
            }

            temp_coefficients[self.n] = self.coefficients[self.n] - learning_rate * self.dJ_db();

            //In Java System.arraycopy(..., ...) or THIS (better than other) :
            std::mem::swap(&mut self.coefficients, &mut temp_coefficients);
        }

        self.coefficients.clone()
    }

    /// Partial derivative of the cost with respect to the `j`-th weight `a_j`:
    ///
    /// ```text
    /// dJ/da_j = (2/m) * sum over i of ( x_j^(i) * ( f(x^(i)) - y^(i) ) )
    /// ```
    ///
    /// `i` walks over the samples while `j` — the feature being derived — stays
    /// fixed. `f(i) - y^(i)` is the signed error of a prediction; multiplying it
    /// by the feature value weights each sample by how much that feature
    /// contributed to the error.
    #[allow(non_snake_case)]
    fn dJ_daj(&self, j : usize) -> f64 {
        let mut result : f64 = 0.0;
        for i in 0..self.m {
            result += self.real_inputs[i][j] * (self.f(i) - self.real_outputs[i]);
        }

        result * 2. / (self.m as f64)
    }

    /// Partial derivative of the cost with respect to the bias `b`:
    ///
    /// ```text
    /// dJ/db = (2/m) * sum over i of ( f(x^(i)) - y^(i) )
    /// ```
    ///
    /// Same shape as [`Self::dJ_daj`] without the feature factor, because `b`
    /// is added to every prediction with a constant weight of 1.
    #[allow(non_snake_case)]
    fn dJ_db(&self) -> f64 {
        let mut result : f64 = 0.0;
        for i in 0..self.m {
            result += self.f(i) - self.real_outputs[i];
        }

        result * 2. / (self.m as f64)
    }

    /// The cost function: the mean squared error of the current coefficients
    /// over the whole training set.
    ///
    /// ```text
    /// J = (1/m) * sum over i of ( f(x^(i)) - y^(i) )^2
    /// ```
    ///
    /// There is no 1/2 factor in front of the mean, which is why the two
    /// derivatives above keep their factor of 2. Squaring makes over- and
    /// under-estimates count the same and punishes large errors harder.
    ///
    /// The sum is accumulated in `f64` and only the finished mean is cast to
    /// `u128`. That order is the whole point: `f64 as u128` saturates, so
    /// casting a residual before squaring it would turn every negative one into
    /// a zero and drop the samples the model under-estimates out of the cost
    /// altogether.
    ///
    /// The cast truncates, so what comes back is the whole part of the cost and
    /// a data set small enough for its mean squared error to fall below 1 reads
    /// as `0`. A diverged run reads as `0` too, for an unrelated reason: the
    /// coefficients are `NaN` by then and `NaN as u128` is `0`. This number
    /// therefore cannot tell a perfect fit from a broken one — the coefficients
    /// can, by coming back `NaN`.
    #[allow(non_snake_case)]
    pub fn J(&self) -> u128 {
        let mut result : f64 = 0.0;
        for i in 0..self.m {
            result += (self.f(i) - self.real_outputs[i]).powi(2);
        }

        (result / (self.m as f64)) as u128
    }

    /// The model's prediction for the sample stored at index `m`.
    ///
    /// Careful: this `m` parameter is a *sample index*, it shadows the field of
    /// the same name that holds the sample count.
    ///
    /// Both branches of the feature-count check are identical for now: the
    /// second one is reserved for a vectorised (SIMD) implementation that should
    /// pay off once there are enough features to amortise its setup cost. The
    /// bias is added once at the end, outside the sum.
    fn f(&self, m : usize) -> f64 {
        let mut result : f64 = 0.0;
        if self.n < 16 {
            for i in 0..self.n {
                result += self.coefficients[i] * self.real_inputs[m][i];
            }
        } else {
            // FixMe : Vectorization
            for i in 0..self.n {
                result += self.coefficients[i] * self.real_inputs[m][i];
            }
        }

        result + self.coefficients[self.n]
    }

    /// Checks that the three vectors describe one consistent data set and
    /// derives the sizes `m` (sample count) and `n` (feature count) from them.
    ///
    /// The arguments are moved in and handed back unchanged so that
    /// [`Self::new`] can build the struct straight from the return value.
    ///
    /// Panics when:
    /// * there are not as many outputs as there are input samples, or
    /// * the coefficient vector is not `n + 1` long — one weight per feature
    ///   plus the bias.
    ///
    /// `n` is read from the first sample only, so rows of differing length are
    /// not caught here; an empty input set panics on the indexing itself.
    fn validate_data_set(
        real_inputs : Vec<Vec<f64>>,
        real_outputs : Vec<f64>,
        initial_coefficients : Vec<f64>
    )
        -> (Vec<Vec<f64>>, Vec<f64>, usize, usize, Vec<f64>)
    {
        let m : usize = real_inputs.len();
        if m != real_outputs.len() {
            panic!("Real Inputs and Output Sample size (m) not equal !!");
        }

        let n : usize = real_inputs[0].len();

        if n + 1 != initial_coefficients.len() {
            panic!("Feature count (n) and a real input sample size (n) not equal !!");
        }

        (real_inputs, real_outputs, m, n, initial_coefficients)
    }

}