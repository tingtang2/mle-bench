# Overview

Can your model unlock the secrets of polymers? In this competition, you're tasked with predicting the fundamental properties of polymers to speed up the development of new materials. Your contributions will help researchers innovate faster, paving the way for more sustainable and biocompatible materials that can positively impact our planet.

## Description

Polymers are the essential building blocks of our world, from the DNA within our bodies to the plastics we use every day. They are key to innovation in critical fields like medicine, electronics, and sustainability. The search for the next generation of groundbreaking, eco-friendly materials is on, and machine learning can be the solution. However, progress has been stalled by one major hurdle: a critical lack of accessible, high-quality data.

Our Open Polymer Prediction 2025 introduces a game-changing, large-scale open-source dataset – ten times larger than any existing resource. We invite you to piece together the missing links and unlock the vast potential of sustainable materials.

Your mission is to predict a polymer's real-world performance directly from its chemical structure. You'll be provided with a polymer's structure as a simple text string (SMILES), and your challenge is to build a model that can accurately forecast five key metrics that determine how it will behave. This includes predicting its density, its response to heat thermal conductivity(Tc) and glass transition temperature(Tg), and its fundamental molecular size and packing efficiency radius of gyration(Rg) and fractional free volume(FFV). The ground truth for this competition is averaged from multiple runs of molecular dynamics simulation.

Your contributions have the potential to redefine polymer discovery, accelerating sustainable polymer research through virtual screening and driving significant advancements in materials science.

## Evaluation

The evaluation metric for this contest is a weighted Mean Absolute Error (wMAE) across five polymer properties, defined as:

wMAE = 1/X * sum(sum(w_i * |y_i - y_pred_i|)) for X in X and i in I(X)

where X is the set of polymers being evaluated, w_i is the reweighting factor for the i-th property, y_i is the true value of the i-th property, and y_pred_i is the predicted value of the i-th property.

To ensure that all property types contribute equally regardless of their scale or frequency, we apply a reweighting factor to each property type:

w_i = (1/r_i) * (K*sqrt(1/n_i))/sum(sqrt(1/n_j)) for j in 1 to K

where n_i is the number of available values for the i-th property, r_i = max(y_i) - min(y_i) is the estimated value range of that property type based on the test data, K is the total number of properties. This design has three goals:

- **Scale normalization**: Division by the range ensures that properties with larger numerical ranges do not dominate the error.
- **Inverse square-root scaling**: To prevent some properties from being overlooked, the term sqrt(1/n_i) assigns a disproportionately high weight to rare properties with fewer samples.
- **Weight normalization**: The second term is normalized so that the sum of weights across all K properties is K.

### Submission File

The submission file for this competition must be a csv file. For each id in the test set, you must predict a value for each of the five chemical properties. The file should contain a header and have the following format.

```
id,density,Tc,Tg,Rg,FFV
1,1.0,1.0,1.0,1.0,1.0
2,2.0,2.0,2.0,2.0,2.0
3,3.0,3.0,3.0,3.0,3.0
```