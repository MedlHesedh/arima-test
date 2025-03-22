import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Set the style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Read the CSV file
file_path = 'materials_prices.csv'
df = pd.read_csv(file_path)

# Convert date to datetime format
df['date'] = pd.to_datetime(df['date'])

# Create the visualization
plt.figure(figsize=(14, 8))

# Plot each material with a different color and line style
materials = df['material_name'].unique()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', '*']

for i, material in enumerate(materials):
    material_data = df[df['material_name'] == material]
    plt.plot(material_data['date'], material_data['price'], 
             label=material, 
             marker=markers[i % len(markers)],
             color=colors[i % len(colors)],
             linewidth=2,
             markersize=8)

# Format the plot
plt.title('Construction Materials Price Trends (2023)', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Format the date on x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

plt.tight_layout()

# Add price annotations
for material in materials:
    material_data = df[df['material_name'] == material]
    last_date = material_data['date'].iloc[-1]
    last_price = material_data['price'].iloc[-1]
    plt.annotate(f'${last_price:.2f}', 
                xy=(last_date, last_price),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold')

# Save the visualization
output_file = 'materials_price_trends.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print(f"Visualization saved as {output_file}")
