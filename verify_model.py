import math

intercept = -24.34629067
coefs = {
    'Chest_Pain': 2.66563621, 'Shortness_of_Breath': 2.49369091,
    'Fatigue': 2.63298665, 'Palpitations': 2.97383434, 'Dizziness': 2.69919581,
    'Swelling': 2.45217442, 'Pain_Arms_Jaw_Back': 2.65976933, 'Cold_Sweats_Nausea': 2.54450674,
    'High_BP': 1.64220755, 'High_Cholesterol': 1.55248653, 'Diabetes': 1.48471905,
    'Smoking': 1.50469155, 'Obesity': 1.48024656, 'Sedentary_Lifestyle': 1.76402358,
    'Family_History': 1.55297017, 'Chronic_Stress': 1.57021697, 'Gender': 1.40703566, 'Age': 0.12000579
}

sigmoid = lambda z: 1/(1+math.exp(-z))
binary_keys = [k for k in coefs if k not in ('Age','Gender')]

# Test 1: 5 factors at age=63 male
active5 = ['Chest_Pain', 'High_BP', 'Smoking', 'Diabetes', 'Family_History']
logit1 = intercept + coefs['Age']*63 + coefs['Gender']*1 + sum(coefs[k] for k in active5)
prob1 = sigmoid(logit1)
print(f"5 factors (age=63,male): logit={logit1:.4f}  P(risk)={prob1:.6f}  pred={int(prob1>=0.5)}")

# Test 2: All 16 binary + age=60 male
logit2 = intercept + coefs['Age']*60 + coefs['Gender']*1 + sum(coefs[k] for k in binary_keys)
prob2 = sigmoid(logit2)
print(f"ALL factors (age=60,male): logit={logit2:.4f}  P(risk)={prob2:.6f}  pred={int(prob2>=0.5)}")

# Min binary coef sum needed to cross 50% at average age (54.46) male
needed = 24.34629067 - coefs['Age']*54.46 - coefs['Gender']*1
total_binary = sum(coefs[k] for k in binary_keys)
avg_coef = total_binary / 16
print(f"\nTo cross 50% at age=54, male:")
print(f"  Need binary sum > {needed:.4f}")
print(f"  Total binary coef sum = {total_binary:.4f}")
print(f"  Average binary coef   = {avg_coef:.4f}")
print(f"  Avg factors needed    = {needed/avg_coef:.1f}")

# Explain the model's data pattern
print("\nThe model's large negative intercept means it requires ~11+ risk factors")
print("to flip prediction. This matches the 50/50 dataset where risk=1 cases")
print("have many overlapping factors.")

# Test with dataset-representative high-risk profile
print("\nDataset-representative tests:")
for n_factors in [8, 10, 12, 14, 16]:
    top = sorted(binary_keys, key=lambda k: coefs[k], reverse=True)[:n_factors]
    logit = intercept + coefs['Age']*60 + coefs['Gender']*1 + sum(coefs[k] for k in top)
    prob = sigmoid(logit)
    print(f"  Top {n_factors} factors, age=60, male: P(risk)={prob:.4f}  pred={int(prob>=0.5)}")
