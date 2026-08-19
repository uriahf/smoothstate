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

cat(sprintf("rms::rcs knot max abs diff:        %.12g\n", knot_diff))
cat(sprintf("rms::cph + rcs curve max abs diff: %.12g\n", rms_curve_diff))

stopifnot(knot_diff < 1e-10)
stopifnot(rms_curve_diff < 1e-5)
