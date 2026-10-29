# Weekly input: calibration (0.1.42)

`scripts/calibrate_weekly.py` measures how often the weekly verdict and the
weekly promotion lift raise a false alarm, and how often they find a real
change, on synthetic weekly sales labelled by the week ending Sunday: a level
of 10,000 a week, growth of 10% a year, an annual swing of none or ±15%, and
noise of 8% a week, independent or AR(1) at 0.4 (sell-out weeks follow each
other). 200 seeds per row, with the Shopify app's calibration (expected band 2,
noise band 2.5, test at 5%, intervals at 90%). Run it with
`PYTHONPATH=. python scripts/calibrate_weekly.py --method fourier` (or `stl`).

## When there's enough history

| | First answers at |
|---|---|
| A week's verdict | 77 weeks of data |
| Four weeks' verdict | 83 weeks |
| A four-week promotion's lift | 83 weeks if it ends at the latest week, 93 if it ends 10 weeks earlier |

Fourier fits the annual pattern from 52 weeks, but the verdict and the lift
rank against the expectation's past errors, which need a seasonal one year
back and 20 past positions. Below these they return `Insufficient`, which the
sales tools word as what is needed. So the tools say weekly data "needs about
18 months of history before a week can be judged, and works best with two
years or more". Two and three years performed about the same.

## What it shows

- **False alarms stay at or under nominal.** With Fourier, a week's verdict
  calls an untouched week a real change 0 to 3% of the time; four weeks, 1 to
  6.5% (the test's nominal rate is 5%, and 200 seeds measure it to about ±1.5
  points). An untouched promotion is called a real change 1.5 to 5% of the
  time, and its 90% interval excludes zero 5 to 11% of the time (nominal 10%).
- **A single week is hard to call.** One week up 20% is found 22 to 48% of the
  time at 8% weekly noise: the verdict needs the week outside both bands and
  the test, by design. Four weeks up 20% are found 76 to 92% of the time, and
  a four-week promotion of +20% 68 to 89% of the time.
- **The lift is about right.** Its mean estimate is +20 to +22% for a true
  +20%, and its 90% interval holds the truth 82 to 88% of the time.
- **Fourier, not STL, for weekly input.** STL needs two full years before it
  answers at all; at 104 weeks its lift finds a +20% promotion 61 to 76% of
  the time against Fourier's 73 to 89%, with more false alarms under
  autocorrelated noise (6% against 4.5 to 5%). At 156 weeks the two are much
  alike. Fourier's reference is exactly out of sample; STL's is in sample.

## Fourier (the default for weekly input)

**The verdict.** False alarm: the share of untouched periods called a real change (the test's nominal rate is 5%, and the bands make the verdict stricter). Power: the share found when every week of the period is +20%.

| Weeks of history | Annual swing | Weekly noise | 1 week: false alarm | 1 week: power | 4 weeks: false alarm | 4 weeks: power |
|---|---|---|---|---|---|---|
| 78 | none | independent | 1.5% | 26.0% | n/a | n/a | 4w TOO_SHORT, needs 79 weeks
| 78 | none | AR(1) 0.4 | 1.5% | 47.0% | n/a | n/a | 4w TOO_SHORT, needs 79 weeks
| 78 | ±15% | independent | 1.5% | 22.5% | n/a | n/a | 4w TOO_SHORT, needs 79 weeks
| 78 | ±15% | AR(1) 0.4 | 1.5% | 36.0% | n/a | n/a | 4w TOO_SHORT, needs 79 weeks
| 104 | none | independent | 2.0% | 31.0% | 5.0% | 88.0% |
| 104 | none | AR(1) 0.4 | 2.5% | 47.0% | 5.5% | 76.0% |
| 104 | ±15% | independent | 2.0% | 33.5% | 4.0% | 90.5% |
| 104 | ±15% | AR(1) 0.4 | 3.0% | 47.5% | 5.0% | 79.5% |
| 156 | none | independent | 2.0% | 28.0% | 3.5% | 89.0% |
| 156 | none | AR(1) 0.4 | 1.0% | 44.0% | 6.5% | 76.5% |
| 156 | ±15% | independent | 1.5% | 31.5% | 1.0% | 91.5% |
| 156 | ±15% | AR(1) 0.4 | 0.0% | 47.5% | 3.0% | 81.0% |

**The promotion lift**, a four-week window. False alarm: an untouched window called a real change. The 90% interval should exclude zero for about 10% of untouched windows. Power: the share found when the four weeks are +20%; then the mean estimate and how often the 90% interval holds +20%.

| Weeks of history | Annual swing | Weekly noise | False alarm | Interval excludes 0 | Power | Mean estimate | Interval holds +20% |
|---|---|---|---|---|---|---|---|
| 104 | none | independent | 2.5% | 9.0% | 88.5% | +21.7% | 84.5% |
| 104 | none | AR(1) 0.4 | 5.0% | 10.0% | 74.5% | +22.2% | 82.5% |
| 104 | ±15% | independent | 1.5% | 8.0% | 86.5% | +21.0% | 86.0% |
| 104 | ±15% | AR(1) 0.4 | 4.5% | 11.0% | 73.0% | +21.5% | 83.0% |
| 156 | none | independent | 3.5% | 8.5% | 87.0% | +20.9% | 86.5% |
| 156 | none | AR(1) 0.4 | 4.0% | 5.0% | 71.0% | +21.1% | 88.5% |
| 156 | ±15% | independent | 3.0% | 7.0% | 84.0% | +20.2% | 85.5% |
| 156 | ±15% | AR(1) 0.4 | 3.0% | 6.5% | 68.0% | +20.3% | 87.0% |

## STL, for comparison

**The verdict.** False alarm: the share of untouched periods called a real change (the test's nominal rate is 5%, and the bands make the verdict stricter). Power: the share found when every week of the period is +20%.

| Weeks of history | Annual swing | Weekly noise | 1 week: false alarm | 1 week: power | 4 weeks: false alarm | 4 weeks: power |
|---|---|---|---|---|---|---|
| 156 | none | independent | 1.5% | 27.5% | 5.5% | 94.5% |
| 156 | none | AR(1) 0.4 | 1.0% | 43.5% | 8.5% | 84.0% |
| 156 | ±15% | independent | 1.0% | 31.0% | 3.0% | 93.5% |
| 156 | ±15% | AR(1) 0.4 | 0.5% | 47.5% | 6.0% | 83.5% |

**The promotion lift**, a four-week window. False alarm: an untouched window called a real change. The 90% interval should exclude zero for about 10% of untouched windows. Power: the share found when the four weeks are +20%; then the mean estimate and how often the 90% interval holds +20%.

| Weeks of history | Annual swing | Weekly noise | False alarm | Interval excludes 0 | Power | Mean estimate | Interval holds +20% |
|---|---|---|---|---|---|---|---|
| 104 | none | independent | 3.0% | 8.0% | 76.0% | +21.9% | 83.5% |
| 104 | none | AR(1) 0.4 | 6.0% | 12.0% | 62.0% | +22.7% | 83.5% |
| 104 | ±15% | independent | 3.0% | 8.5% | 75.5% | +21.6% | 84.0% |
| 104 | ±15% | AR(1) 0.4 | 6.0% | 11.5% | 60.5% | +22.4% | 84.0% |
| 156 | none | independent | 4.0% | 13.0% | 88.0% | +20.9% | 79.5% |
| 156 | none | AR(1) 0.4 | 4.5% | 11.0% | 78.5% | +21.1% | 82.5% |
| 156 | ±15% | independent | 3.5% | 12.0% | 86.0% | +20.2% | 79.0% |
| 156 | ±15% | AR(1) 0.4 | 4.5% | 11.0% | 77.0% | +20.5% | 82.5% |

