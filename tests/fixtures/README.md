# Test fixtures

`demo_store_daily.csv` — 742 daily values (Mon 16 Sep 2024 to Sun 27 Sep 2026)
from a generated demo store (seed 1). Money columns are integer minor units.
`orders` is the number of orders behind each day's money figures.

What was planted, measured on these values rather than taken from the
generator's parameters:

- **November 2025**: mean daily `net_revenue` ÷ its centred 365-day moving
  average − 1 is +31.28 pt including the Black Friday weekend and Cyber Monday,
  +18.94 pt without.
- **Promotion** 14–24 May 2026: against the 28 days before, revenue per day
  +24.7%, contribution margin per day −18.7%.
- **Acquisition decline** in the last four weeks (31 Aug – 27 Sep 2026): new
  customers per week 119, 94, 77, 55 against 151, 157, 146, 139 the four
  weeks before; returning customers hold.
- Also present: new-customer arrivals rising 40% over two years, a weekly
  pattern busiest at weekends, Black Friday in 2024 and 2025, a post-promotion
  dip from pull-forward.
