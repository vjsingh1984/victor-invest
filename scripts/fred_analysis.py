import pandas_datareader.data as fred
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =================================================================
# 1. Define FRED Series IDs
# =================================================================
series_ids = {
    "DowJones_US_TMI": "DJUSTCD",
    "Real_GDP": "GDPC1",
    "Federal_Debt": "GFDEBTN",
    "Household_Debt": "BOGZ1FL194190005Q",
}

# =================================================================
# 2. Download Data
# =================================================================
# Note: You must have an internet connection and the 'pandas_datareader' library installed.
df = None
try:
    print("Downloading data from FRED...")
    df = fred.DataReader(list(series_ids.values()), "fred", start="1970-01-01")
    df.columns = series_ids.keys()
except Exception as e:
    print(f"Error downloading data: {e}")
    print("Please ensure you have 'pandas_datareader' installed and a stable internet connection.")
    # Exit or handle error gracefully

# =================================================================
# 3. Data Processing and Alignment
# =================================================================

# Resample the Daily stock index (DJUSTC) to a Quarterly frequency (Quarter-End)
df["DowJones_US_TMI"] = df["DowJones_US_TMI"].resample("Q").last()

# Align all data to the Quarterly frequency and forward-fill any small gaps,
# then drop any rows that still have missing data from the start of the time series.
df_quarterly = df.resample("Q").last()
df_quarterly = df_quarterly.ffill()
df_quarterly.dropna(inplace=True)

# Unit Conversion: Federal Debt (GFDEBTN) is in Millions of Dollars.
# Convert it to Billions of Dollars to match the units of Real GDP.
df_quarterly["Federal_Debt_Adj"] = df_quarterly["Federal_Debt"] / 1000

# =================================================================
# 4. Calculate Ratios to Real GDP
# =================================================================
gdp = df_quarterly["Real_GDP"]
ratios_to_calculate = {
    "Dow Jones US TMI / Real GDP": "DowJones_US_TMI",
    "Federal Debt / Real GDP": "Federal_Debt_Adj",
    "Household Debt / Real GDP": "Household_Debt",
}

ratio_data = {}
for new_name, col_name in ratios_to_calculate.items():
    # Ratio is calculated as (Numerator / GDP) * 100 for percentage
    ratio_data[new_name] = (df_quarterly[col_name] / gdp) * 100

df_ratios = pd.DataFrame(ratio_data)

# =================================================================
# 5. Plotting with Matplotlib and Trendlines
# =================================================================
fig, ax = plt.subplots(figsize=(14, 8))
# Use the 'tab10' colormap for distinct colors
colors = plt.cm.get_cmap("tab10", len(df_ratios.columns))

# Plot each ratio and its linear trendline
for i, col in enumerate(df_ratios.columns):
    color = colors(i)

    # Plot the main data line (Solid Line)
    ax.plot(df_ratios.index, df_ratios[col], label=col, color=color, linewidth=2)

    # Calculate and plot the linear trendline (Dotted Line, Same Color)
    # Prepare data for Linear Regression
    X = np.arange(len(df_ratios)).reshape(-1, 1)
    y = df_ratios[col].values
    reg = LinearRegression().fit(X, y)
    trendline = reg.predict(X)

    # Plot the trendline
    ax.plot(
        df_ratios.index,
        trendline,
        color=color,
        linestyle=":",
        linewidth=1.5,
        label=f'Trendline ({col.split(" / ")[0]})',
    )

# Add labels and title
ax.set_title("Selected US Economic Indicators as a Percentage of Real GDP", fontsize=16)
ax.set_xlabel("Date (Quarterly)", fontsize=12)
ax.set_ylabel("Ratio to Real GDP (%)", fontsize=12)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)

# Save the figure
plot_filename = "fred_economic_ratios_vs_gdp.png"
plt.savefig(plot_filename)
print(f"\nSuccessfully generated and saved plot to {plot_filename}")

# Display a sample of the calculated ratios
print("\nSample of Calculated Ratios (Percent of Real GDP):")
print(df_ratios.tail())
