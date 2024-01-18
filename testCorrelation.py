import json
import matplotlib.pyplot as plt
import numpy as np

# Load data from the JSON files
with open('saved_individuals_11.json', 'r') as backward_file:
    backward_data = json.load(backward_file)

with open('test_individuals_11.json', 'r') as forward_file:
    forward_data = json.load(forward_file)

# Extract fitness scores and profits
fitness_scores = [individual['fitness'] for individual in backward_data]
profits = [result['Profit'] for result in forward_data]

# Create a scatter plot with red dots
plt.figure(figsize=(8, 6))
plt.scatter(fitness_scores, profits, color='red', alpha=0.7, label='Data')

# Calculate and plot the line of best fit
fit = np.polyfit(fitness_scores, profits, 1)
fit_fn = np.poly1d(fit)
plt.plot(fitness_scores, fit_fn(fitness_scores), color='blue', label='Line of Best Fit')

# Calculate R-squared value (goodness of fit)
correlation_matrix = np.corrcoef(fitness_scores, profits)
correlation_xy = correlation_matrix[0, 1]
r_squared = correlation_xy ** 2

# Add R-squared value to the plot
plt.text(min(fitness_scores), max(profits), f'R-squared: {r_squared:.2f}', fontsize=10)

# Plot settings
plt.title('Fitness vs. Profit Scatter Plot with Line of Best Fit')
plt.xlabel('Fitness Scores (Backward Testing)')
plt.ylabel('Profits (Forward Testing)')
plt.legend()
plt.grid(True)
plt.show()
