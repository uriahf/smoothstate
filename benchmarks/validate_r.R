#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(survival)
  library(rms)
})

input <- read.csv("benchmark-results/r-validation-input.csv")
grid <- read.csv("benchmark-results/r-validation-grid.csv")
reference <- read.csv("benchmark-results/r-validation-reference.csv")
ref <- setNames(reference$value, reference$quantity)
horizon <- unname(ref[["horizon"]])

# 1) Canonical survival::coxph validation on the exact spline columns used by smoothstate.
fit_survival <- coxph(
  Surv(time, event) ~ rcs_1 + rcs_2,
  data = input,
  ties = "efron",
  x = TRUE,
  y = TRUE
)

coef_diff <- max(abs(unname(coef(fit_survival)) - c(ref[["beta_1"]], ref[["beta_2"]])))
bh <- basehaz(fit_survival, centered = FALSE)
h0_survival <- if (any(bh$time <= horizon)) max(bh$hazard[bh$time <= horizon]) else 0
lp_survival <- as.matrix(grid[, c("rcs_1", "rcs_2")]) %*% unname(coef(fit_survival))
risk_survival <- 1 - exp(-h0_survival * exp(lp_survival))
survival_curve_diff <- max(abs(as.numeric(risk_survival) - grid$smoothstate_risk))

# 2) Harrell rms validation using the natural rcs(x, 3) specification.
# datadist() needs the named data frame so rms can resolve it later.
dd <- datadist(input)
options(datadist = "dd")
fit_rms <- cph(
  Surv(time, event) ~ rcs(x, 3),
  data = input,
  method = "efron",
  x = TRUE,
  y = TRUE,
  surv = TRUE
)

rms_knots <- fit_rms$Design$parms$x
knot_diff <- max(abs(as.numeric(rms_knots) - c(ref[["knot_1"]], ref[["knot_2"]], ref[["knot_3"]])))

rms_surv <- survest(
  fit_rms,
  newdata = data.frame(x = grid$x),
  times = horizon,
  conf.int = 0
)$surv
risk_rms <- 1 - as.numeric(rms_surv)
rms_curve_diff <- max(abs(risk_rms - grid$smoothstate_risk))

cat(sprintf("survival::coxph coefficient max abs diff: %.12g\n", coef_diff))
cat(sprintf("survival::coxph curve max abs diff:       %.12g\n", survival_curve_diff))
cat(sprintf("rms::rcs knot max abs diff:               %.12g\n", knot_diff))
cat(sprintf("rms::cph + rcs curve max abs diff:        %.12g\n", rms_curve_diff))

stopifnot(coef_diff < 1e-6)
stopifnot(survival_curve_diff < 1e-6)
stopifnot(knot_diff < 1e-10)
stopifnot(rms_curve_diff < 1e-5)
