# Monoclonal Antibody Production Optimization

Simulated monoclonal antibody production data is used to recommend initial glucose, glutamine, viable cell density, and harvest time. It compares polynomial and k-nearest neighbours models, forecasts complete batches, and ranks operating conditions by space-time profit. The workflow uses 30 batch datasets and reports estimated production value, profit, yield, and 95% prediction intervals calculated through simulation.

## Highlights

- Selected 30 batch runs using a three-level factorial design with repeat runs
- Built and compared polynomial and k-nearest neighbours models for batch forecasting
- Evaluated forecasts using nested cross-validation that keeps complete batches together
- Optimized starting conditions and harvest time, and estimated prediction intervals through residual simulations

## Environment

- Python: 3.14.7
- NumPy: 2.5.2
- pandas: 3.0.5
- scikit-learn: 1.9.0

## Code Design

### Loading and organizing the data

Running `main.py` starts the workflow and calls the functions in `src/functions.py`. The program locates the project directory, reads `initial_conditions.csv`, and loads the 30 selected batch files from `datasets`. Each filename supplies its batch ID, which connects that run to its nominal starting conditions. These starting conditions are stored separately from the changing measurements because they determine the culture cost. The combined table contains 2550 measurements and time is converted from hours to days so that the economic calculations have consistent units.

### Calculating economic performance

The program calculates culture cost from initial glucose, glutamine, and viable cell density. At each measurement time, it calculates culture value from mAb concentration and subtracts the culture cost to obtain profit. Profit and mAb concentration are divided by harvest time plus one day of downtime to obtain space-time profit and space-time yield. The best recorded harvest time for each batch is then exported to `batch_summary.csv`.

### Preparing and training the models

For each batch, the program pairs the seven process measurements at one time with the same seven variables eight hours later. The inputs contain glucose, glutamine, lactate, ammonia, viable cell density, total cell density, and mAb concentration. Each batch provides 84 pairs which give 2,520 training examples. Then, pairs are created within individual batches so that the end of one run is never connected to the start of another. 

Both regression pipelines used standardize the inputs. Polynomial regression tests degrees one and two, while k-nearest neighbours tests two, five, and ten neighbours with uniform weights. The five-fold grouped cross-validation selects hyperparameters using mAb prediction RMSE. This focuses scoring on the product of interest rather than combining errors from variables with different units, and then the selected pipelines are fitted using all 30 runs.

### Validating complete batch forecasts

To validate, a second outer five-fold validation checks complete forecasts with each outer fold holding out six batches. Both models are tuned again using only the remaining 24 batches, keeping the outer test data separate from hyperparameter selection. Forecasts start from the known initial nutrient and cell concentrations, with zero initial lactate, ammonia, and mAb. The initial total cell density equals initial viable cell density. Each predicted state becomes the input for the next step and forecasts are limited to nonnegative values and each variable's observed training maximum. The program compares full forecast errors in mAb concentration and space-time profit, selecting the model with lower profit RMSE.

### Searching operating conditions

The optimization stage extracts 343 unique starting-condition combinations from the initial-condition file. It forecasts every combination over the 85 available time points, covering zero to 28 days at eight-hour intervals. This uses predictions from the fitted model rather than additional batch datasets and for each combination, the program calculates culture cost and the predicted profit and yield at every possible harvest time. It retains the harvest time with the highest space-time profit and ranks the resulting 343 candidates. This connects the biological forecast to the economic objective.

### Estimating uncertainty and exporting results

The program estimates uncertainty at the recommended conditions using 1000 simulated trajectories. It calculates and centres the fitted model's one-step residual errors, then samples complete seven-variable residual vectors during forecasting. The sampling vectors preserves relationships between the variable errors, and predictions are bounded before adding noise, while negative states are prevented after noise is added. A fixed random seed makes the simulations reproducible. The 2.5th and 97.5th percentiles provide nominal 95% prediction intervals, which are converted into value, profit, and yield intervals using the same economic equations.

The workflow exports CSV tables to `tables`, including model comparisons, validation forecasts, optimization results, and prediction intervals. These tables provide the numerical evidence used below. The program also prints its results and environment versions so that the recommendation and software requirements can be checked after execution.

## Analysis

### Selection of the 30 batch runs

The run selection was based on the initial-condition table before examining production outcomes. A three-level factorial design was used, with low, middle, and high values for each starting variable:

| Initial condition | Low | Middle | High |
|---|---:|---:|---:|
| Glucose [mM] | 5 | 20 | 35 |
| Glutamine [mM] | 1 | 4 | 7 |
| VCD [cells/mL] | 200,000 | 800,000 | 1,400,000 |

One run was selected for each combination, giving 27 different conditions. Three additional runs repeated the all-low, all-middle, and all-high conditions. The repeated pairs were batches 1 and 2, 514 and 515, as well as 1027 and 1028. All 30 runs contributed to model fitting and validation.

This design covers the boundaries and middle of the available ranges while including combinations in which the inputs change independently. It gives the models examples of nutrient and cell-density interactions instead of concentrating the run budget around one region. The additional repeats allow some comparison of variations with runs at the same settings. 

The design also leaves many starting conditions unmeasured. For example, the recommended initial VCD of 600,000 cells/mL is between the selected low and middle levels, and predictions at these settings would need interpolation. Choosing runs before inspecting their mAb outcomes avoids selecting only favourable examples, but the resulting coverage still limits the strength of the final recommendation.

### Model architecture and hyperparameters

Both alternatives use nonlinear vector autoregression with one previous state, or NVAR(1). Each model predicts the complete seven-variable state eight hours ahead, including nutrients, cells, and byproducts allows different initial conditions to produce different trajectories. The architecture assumes that the current measured state has enough information to predict the next state. Using one lag also avoids needing unavailable measurements before the start of a batch.

Polynomial regression can describe curvature and interactions between process variables. Degree one provides a simpler candidate, while degree two adds squared and interaction terms. With seven input variables, degree two produces 35 polynomial features before the fitted intercept. Cross-validation selected degree two, supporting the additional flexibility within the tested settings. 

The alternative k-nearest neighbours model predicts by averaging the next states associated with similar current states. It provides a different way to represent nonlinear behaviour without specifying a polynomial relationship. The tested neighbourhood sizes were two, five, and ten, with ten neighbours giving the lowest one-step validation error. Uniform weighting averaged neighbouring observations equally. Standardization is important for this model because cell densities would otherwise dominate distances when measured with much smaller concentration values.

### Model comparison and validation

The comparison results are exported in [model_comparison.csv](tables/model_comparison.csv) and [forecast_comparison.csv](tables/forecast_comparison.csv).

| Model | Selected setting | One-step mAb RMSE [mg/L] | Full forecast mAb RMSE [mg/L] | Full forecast space-time profit RMSE [$/L/d] |
|---|---|---:|---:|---:|
| Polynomial | Degree 2 | 0.056149 | 0.255968 | 24.380296 |
| kNN | 10 neighbours | 0.096118 | 0.385655 | 41.785973 |

The full forecast columns evaluate the complete modelling procedures with hyperparameters selected again inside each outer training fold. They are therefore not tests of one fixed model fitted to all batches. The selected-setting column describes the final models trained using the full dataset.

Polynomial regression had lower error for both complete mAb forecasts and space-time profit. Its profit RMSE was approximately 42% lower than the kNN result, making it the selected model for optimization. Profit error was the main selection criterion because the production objective includes both culture cost and cycle duration.

Full forecast errors were higher than one-step errors. During a complete forecast, predicted states become future inputs, which can cause errors to increase. This makes the complete forecast test more relevant to recommending conditions before a batch starts. The individual held-out predictions are available in [validation_forecasts.csv](tables/validation_forecasts.csv).

### Influence of initial glucose

Input effects were examined using [optimization_results.csv](tables/optimization_results.csv). For each comparison below, the other two starting conditions were held at their recommended values. Each candidate used its own harvest time that maximized predicted space-time profit and the reported yields are therefore yields at those profit-selected harvest times.

At 1 mM glutamine and 600,000 cells/mL initial VCD, increasing glucose from 5 to 20 mM raised predicted space-time profit from 39.92 to 87.00/(L·d). Space-time yield increased from 0.1052 to 0.1910 mg/(L·d). Increasing glucose to 35 mM reduced profit to 66.19/(L·d) and yield to 0.1619 mg/(L·d).

The model therefore favours an intermediate starting glucose concentration under these conditions. Higher glucose also increases the specified culture cost by $10/L for every additional mM. The predicted production benefit has to compensate for this cost and any change in the best harvest time. 

### Influence of initial glutamine

At 20 mM glucose and 600,000 cells/mL initial VCD, 1 mM glutamine gave the highest predicted profit of 87.00/(L·d). Increasing glutamine to 3 mM increased the yield at the selected harvest time from 0.1910 to 0.2071 mg/(L·d), but profit decreased to 77.82/(L·d). At 7 mM, profit and yield decreased even more to 48.19/(L·d) and 0.1282 mg/(L·d), respectively.

This shows the difference between productivity and profitability. Additional glutamine costs $20/L per mM, and its production benefit needs to be evaluated together with cycle time. The highest yield among these glutamine settings does not provide the highest profit. Since 1 mM is the lower boundary of the search, the result supports that value within the range but doesnt determine if a lower concentration would perform better.

### Influence of initial viable cell density

At 20 mM glucose and 1 mM glutamine, increasing initial VCD from 200,000 to 600,000 cells/mL raised profit from 77.88 to 87.00/(L·d). Increasing it to 1,000,000 cells/mL raised yield to 0.2198 mg/(L·d), but profit dropped to 75.79/(L·d). At 1,400,000 cells/mL, profit was $55.20/(L·d).

Each additional 200,000 cells/mL increases the specified culture cost by 100/L. Higher starting cell density can therefore improve predicted output while lowering the economic benefit. The recommended density balances these effects under the supplied cost assumptions. The 400,000 cells/mL option also performed similarly, with predicted profit of 86.09/(L·d), making it another useful condition for future comparison.

### Recommended conditions, harvest time, and uncertainty

The highest predicted space-time profit among the 343 starting-condition combinations and 85 harvest times occurred at the following settings:

| Quantity | Recommendation or point estimate | Nominal 95% prediction interval |
|---|---:|---:|
| Initial glucose | 20 mM | — |
| Initial glutamine | 1 mM | — |
| Initial VCD | 600,000 cells/mL | — |
| Harvest time | 4 d | — |
| Culture cost | $520.00/L | — |
| mAb concentration at harvest | 0.9550 mg/L | 0.7185–1.1898 mg/L |
| Culture value at harvest | $955.02/L | $718.51–$1,189.77/L |
| Space-time profit | $87.00/(L·d) | $39.70–$133.95/(L·d) |
| Space-time yield | 0.1910 mg/(L·d) | 0.1437–0.2380 mg/(L·d) |

Culture cost is 200/L for glucose, 20/L for glutamine, and 300/L for the initial cells. The predicted culture value gives a batch profit of $435.02/L. Dividing this by four production days plus one downtime day gives the reported space-time profit, while cost is fixed by the chosen initial conditions.

Harvesting is selected using profit per cycle day rather than final product concentration alone. [recommended_forecast.csv](tables/recommended_forecast.csv) shows how additional product accumulation and longer cycle duration influence the simulated trajectories. Point forecasts determine the recommended settings, while separate simulations estimate their uncertainty. The resulting intervals are exported in [prediction_intervals.csv](tables/prediction_intervals.csv).

The best recorded run in [batch_summary.csv](tables/batch_summary.csv) was batch 515, which reached $92.31/(L·d) at approximately 3.33 days with 20 mM glucose, 4 mM glutamine, and 800,000 cells/mL initial VCD. That recorded value exceeds the recommended candidate's point prediction. The recommendation is therefore a model-generated candidate for further validation, and these results do not show an improvement over the best observed run.

### Advantages and limitations

The workflow compares two regression approaches, separates batches during validation, and evaluates forecasts using the economic objective. Polynomial regression provides a compact model that can represent interactions. The simulation stage also reports a range of possible outcomes instead of treating the point forecast as certain.

There are important limits to this evidence though. Only 30 runs were available, and some held-out runs have a training replicate with identical starting conditions. The validation therefore assesses new runs more directly than entirely unseen operating regions. 

The nominal 95% intervals assume that the centred residual vectors represent future errors and can be sampled independently over time. They preserve correlations between variable errors within a sampled vector, but don't include parameter uncertainty, structural model error, or uncertainty introduced by selecting the maximum from many candidates. Their actual 95% coverage wasn't established on independent batches. Finally, the search uses discrete starting levels and eight-hour harvest intervals, so the recommendation is the best candidate on that grid rather than a proven continuous optimum.
