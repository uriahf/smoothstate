#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(survival)
})

input <- read.csv("benchmark-results/r-validation-input.csv")
grid <- read.csv("benchmark-results/r-validation-grid.csv")
reference <- read.csv("benchmark-results/r-validation-reference.csv")
ref <- setNames(reference$value, reference$quantity)
horizon <- unname(ref[["horizon"]])

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

cat(sprintf("survival::coxph coefficient max abs diff: %.12g\n", coef_diff))
cat(sprintf("survival::coxph curve max abs diff:       %.12g\n", survival_curve_diff))

stopifnot(coef_diff < 1e-6)
stopifnot(survival_curve_diff < 1e-6)
