import matplotlib.pyplot as plt
import json
import numpy as np

with open(r'd:\disease prediction\model_weights.json', 'r') as f:
    data = json.load(f)

metrics = data['metrics']
cm = data['confusion_matrix']

# Accuracy Graph
labels = ['Train Accuracy', 'Test Accuracy', 'CV Score']
values = [metrics['train_accuracy'], metrics['test_accuracy'], metrics['cv_score']]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color=['#4CAF50', '#2196F3', '#FFC107'])
plt.ylim(0.98, 1.0)
plt.ylabel('Accuracy')
plt.title('Overall Accuracy Metrics')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0005, f'{yval:.4f}', ha='center', va='bottom')
plt.savefig(r'd:\disease prediction\accuracy_graph.png')
plt.close()

# Confusion Matrix
cm_matrix = np.array([[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]])
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm_matrix, cmap='Blues')
plt.title('Confusion Matrix', pad=20)
fig.colorbar(cax)

for (i, j), z in np.ndenumerate(cm_matrix):
    ax.text(j, i, '{:0,d}'.format(z), ha='center', va='center')

ax.set_xticklabels([''] + ['Negative', 'Positive'])
ax.set_yticklabels([''] + ['Negative', 'Positive'])
ax.xaxis.set_ticks_position('bottom')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(r'd:\disease prediction\confusion_matrix.png')
plt.close()

print("Graphs generated.")
