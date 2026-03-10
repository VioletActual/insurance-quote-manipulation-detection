# Detecting Quote Manipulation in Motor Insurance

## Overview

Customers requesting insurance quotes may repeatedly modify inputs such as mileage or driving history to reduce the quoted premium. This behaviour, often called *quote manipulation* or *quote gaming*, can lead to mispriced insurance policies.

This project explores whether behavioural patterns in quoting activity can help identify suspicious quote behaviour using machine learning.

The goal is not to build a production system, but to explore how behavioural analytics could support insurance pricing decisions.

---

## Why I Built This Project

Real insurance datasets are rarely public. To learn how pricing and risk modelling works in practice, I simulated quote behaviour on top of a public insurance dataset.

Simulation allows experimentation with modelling techniques, behavioural feature engineering, and model monitoring concepts that would normally require proprietary industry data.

---

## Basic Insurance Terms

**Insurance**  
A financial agreement where customers pay a *premium* to transfer financial risk to an insurer.

**Premium**  
The price paid for insurance coverage.

**Policy**  
The insurance contract between the insurer and the customer.

**Quote**  
A price estimate provided before purchasing an insurance policy.

**Claim**  
A request by the customer for compensation after a loss event (for example a car accident).

**Underwriting**  
The process insurers use to assess risk and determine the appropriate premium.

---

## Why Quote Manipulation Matters

Insurance pricing assumes customers provide accurate information.

If customers repeatedly adjust inputs across multiple quotes to reduce the premium, insurers may unknowingly price policies below the true level of risk.

Over time this can lead to:

- mispriced policies  
- increased claim losses  
- reduced profitability  

Detecting suspicious quote behaviour helps maintain pricing integrity.

---

## Connection to External Risk Events

External factors such as **natural disasters, economic shocks, or large-scale accidents** can significantly increase insurance claims.

During periods of elevated risk (for example storms or floods), insurers need accurate pricing and risk assessment more than ever.

Identifying suspicious quoting behaviour helps reduce the risk of underpriced policies and supports better risk management during periods when claim volumes may increase.

---

## Project Workflow

1. Start with a public synthetic motor insurance dataset  
2. Simulate realistic customer quote histories  
3. Engineer behavioural features from quote activity  
4. Train predictive models to detect suspicious patterns  
5. Evaluate models using cross-validation  
6. Investigate candidate behavioural rating factors  
7. Simulate behaviour drift to test model robustness  
8. Build a simple risk-based intervention framework  

---

## Models Used

- Rule-based baseline detection  
- Random Forest classifier  
- Isolation Forest anomaly detection  

Random Forest produced the strongest performance.

**Cross-validation results**

- Precision ≈ 0.80  
- Recall ≈ 0.80  
- F1 Score ≈ 0.80  

---

## What the Models Learned

Behavioural features revealed several patterns associated with suspicious quote activity.

Important signals include:

- **Short time gaps between quotes** – rapid quote attempts may indicate experimentation with inputs.
- **Repeated premium-lowering edits** – systematic changes that consistently reduce the premium.
- **Multiple quote attempts** within short periods.
- **Large premium drops across quote attempts**.

These findings suggest that quote manipulation is better detected through **behaviour across multiple quote attempts**, rather than through any single input change.

---

## Behavioural Monitoring

A shifted environment was simulated where customers quote more frequently and with shorter time gaps.

Under this scenario:

- precision decreases  
- recall increases  
- overall F1 score declines  

This demonstrates how **behavioural drift can affect model performance**, highlighting the importance of monitoring models in real-world systems.

---

## Risk-Based Intervention Framework

Instead of binary predictions, model outputs were converted into risk scores.

These scores were mapped to operational actions such as:

- No action  
- Prompt customer to confirm details  
- Request additional verification  
- Manual review  

This illustrates how predictive models can support operational decision systems rather than acting as standalone classifiers.

---

## Model Considerations

Several modelling considerations were explored during development.

**Overfitting**  
Models can sometimes learn patterns that exist only in training data. Cross-validation and out-of-fold predictions were used to reduce this risk.

**Outliers**  
Extreme quoting behaviour (for example very large premium drops or unusually frequent quoting) may influence model behaviour. These cases may represent genuine anomalies or rare but legitimate behaviour.

**Model Generalisation**  
The behavioural patterns discovered here are based on simulated quote activity. Real customer behaviour may be more complex and models would require continuous monitoring and retraining.

---

## Limitations

This project uses simulated quote behaviour built on a public synthetic dataset.

Real insurance systems would involve:

- significantly larger datasets  
- richer behavioural signals  
- real customer quote streams  

The simulation was used to explore modelling techniques and behavioural analytics in a controlled environment.

---

## Running the Project

### Requirements

Python libraries used:
pandas
numpy
matplotlib
scikit-learn
jupyter


### Environment

The notebook was developed using **Google Colab**, but it can also be run locally using Jupyter Notebook.

### Steps

1. Clone the repository  
2. Install dependencies from `requirements.txt`  
3. Open `QuoteManipulation.ipynb` in Jupyter or Colab  
4. Run all cells sequentially  

---

## What I Learned

This project was built as a hands-on exploration of behavioural analytics in insurance pricing.

Key takeaways include:

- behavioural features can be more informative than raw customer attributes  
- model evaluation must consider realistic validation strategies  
- monitoring model performance under behavioural drift is important in real systems  
- predictive models are most useful when translated into operational decision frameworks  

Building this project also reinforced the importance of clear explanations, reproducible workflows, and structured experimentation when working on applied machine learning problems.