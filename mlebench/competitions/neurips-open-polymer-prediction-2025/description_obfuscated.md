# Task

Predict five key polymer properties (density, thermal conductivity (Tc), glass transition temperature (Tg), radius of gyration (Rg), and fractional free volume (FFV)) from polymer chemical structures provided as SMILES strings.

# Metric

Weighted Mean Absolute Error (wMAE) across five polymer properties, defined as:

wMAE = 1/X * sum(sum(w_i * |y_i - y_pred_i|)) for X in X and i in I(X)

where X is the set of polymers being evaluated, w_i is the reweighting factor for the i-th property, y_i is the true value of the i-th property, and y_pred_i is the predicted value of the i-th property.

The reweighting factor is calculated as:

w_i = (1/r_i) * (K*sqrt(1/n_i))/sum(sqrt(1/n_j)) for j in 1 to K

where:
- n_i is the number of available values for the i-th property
- r_i = max(y_i) - min(y_i) is the estimated value range of that property type based on the test data
- K is the total number of properties (5)

This weighting ensures:
- **Scale normalization**: Properties with larger numerical ranges do not dominate the error
- **Inverse square-root scaling**: Rare properties with fewer samples receive disproportionately high weight
- **Weight normalization**: The sum of weights across all K properties is K

# Submission Format

For each id in the test set, you must predict a value for each of the five chemical properties. The file should contain a header and have the following format:

```
id,density,Tc,Tg,Rg,FFV
1,1.0,1.0,1.0,1.0,1.0
2,2.0,2.0,2.0,2.0,2.0
3,3.0,3.0,3.0,3.0,3.0
```

# Dataset

The dataset provides polymer structures as SMILES strings and contains five target properties:
- **density**: Polymer density
- **Tc**: Thermal conductivity
- **Tg**: Glass transition temperature
- **Rg**: Radius of gyration
- **FFV**: Fractional free volume

Ground truth values are averaged from multiple runs of molecular dynamics simulation.

