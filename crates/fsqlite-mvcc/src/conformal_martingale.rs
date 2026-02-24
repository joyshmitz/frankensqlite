//! Conformal Martingale Regime Shift Detector (C-MRSD).
//!
//! A distribution-free regime shift detector using conformal p-values and
//! game-theoretic betting martingales (e-processes). Provides finite-sample
//! guarantees via Ville's Inequality.
//!
//! "Distribution-free" here means valid type-I error control under the null
//! assumption of exchangeability. If the underlying data is strongly 
//! non-stationary even within a "regime", the p-values may not be uniform
//! and the detector could trip more often. Power and behavior depend on the
//! chosen non-conformity measure.

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalMartingaleConfig {
    /// Window size for conformal calibration (history).
    pub window_size: usize,
    /// Significance level (alpha). Threshold is 1 / alpha.
    pub alpha: f64,
    /// Kelly betting fraction (lambda), typically in (0, 2).
    pub lambda: f64,
}

impl Default for ConformalMartingaleConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            alpha: 0.05, // Threshold = 20
            lambda: 0.5,
        }
    }
}

pub struct ConformalMartingaleMonitor {
    config: ConformalMartingaleConfig,
    history: VecDeque<f64>,
    wealth: f64,
    observation_count: u64,
    last_change_point: bool,
    regime_mean: f64,
    regime_length: usize,
}

impl ConformalMartingaleMonitor {
    pub fn new(config: ConformalMartingaleConfig) -> Self {
        Self {
            config,
            history: VecDeque::with_capacity(config.window_size),
            wealth: 1.0,
            observation_count: 0,
            last_change_point: false,
            regime_mean: 0.0,
            regime_length: 0,
        }
    }

    pub fn observe(&mut self, x: f64) {
        self.observation_count += 1;
        self.last_change_point = false;

        // Update regime stats
        self.regime_length += 1;
        self.regime_mean += (x - self.regime_mean) / (self.regime_length as f64);

        if self.history.len() < 10 {
            // Need a minimum history to compute meaningful p-values
            if self.history.len() == self.config.window_size {
                self.history.pop_front();
            }
            self.history.push_back(x);
            return;
        }

        // Conformal scoring: rank `x` against history.
        // We use absolute deviation from history median as our non-conformity measure.
        let mut sorted_hist = self.history.clone().into_iter().collect::<Vec<_>>();
        sorted_hist.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted_hist[sorted_hist.len() / 2];

        let score = (x - median).abs();

        let mut greater_or_equal = 1.0; // including x itself
        for &h in &self.history {
            if (h - median).abs() >= score {
                greater_or_equal += 1.0;
            }
        }

        // p-value: fraction of data exchangeable with x
        let p_value = greater_or_equal / (self.history.len() as f64 + 1.0);

        // Update betting martingale
        // We use a fixed betting strategy $f(p) = 1 + \lambda (0.5 - p)$.
        // Because $p$ is uniformly distributed on [0,1] under the null,
        // $E[p] = 0.5$, so $E[f(p)] = 1.0$, making the wealth process a valid martingale.
        // \lambda serves as the betting strategy parameter (Kelly fraction).
        let f_p = self.config.lambda.mul_add(0.5 - p_value, 1.0);
        self.wealth *= f_p.max(0.01); // Prevent total ruin

        // Check Ville's Inequality
        let threshold = 1.0 / self.config.alpha;
        if self.wealth > threshold {
            self.last_change_point = true;
            // Reset martingale and regime stats
            self.wealth = 1.0;
            self.regime_mean = x;
            self.regime_length = 1;
            // Clear history to adapt immediately to the new regime
            self.history.clear();
        }

        if self.history.len() == self.config.window_size {
            self.history.pop_front();
        }
        self.history.push_back(x);
    }

    pub fn change_point_detected(&self) -> bool {
        self.last_change_point
    }

    pub fn current_regime_stats(&self) -> crate::bocpd::RegimeStats {
        crate::bocpd::RegimeStats {
            mean: self.regime_mean,
            length: self.regime_length,
        }
    }

    pub fn observation_count(&self) -> u64 {
        self.observation_count
    }

    pub fn current_wealth(&self) -> f64 {
        self.wealth
    }
}
