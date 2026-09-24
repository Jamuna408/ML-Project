# ============================================================
# TRAVEL TIME PREDICTION USING RIDGE AND LASSO REGRESSION
# DATA ANALYTICS + EDA (FULL SCRIPT)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ============================================================
# 1. UPLOAD DATASET
# ============================================================

from google.colab import files

uploaded = files.upload()

filename = next(iter(uploaded))
df = pd.read_csv(filename)

# ============================================================
# DATA ANALYTICS
# ============================================================

# 2. DATASET

print("DATASET")
print(df.head())


# 3. SHAPE

print("\nSHAPE")
print(df.shape)


# 4. COLUMN NAMES

print("\nCOLUMN NAMES")
print(df.columns.tolist())


# 5. DATA TYPES

print("\nDATA TYPES")
print(df.dtypes)


# 6. NULL VALUES

print("\nNULL VALUES")
print(df.isnull().sum())


# 7. TOTAL NULL VALUES

print("\nTOTAL NULL VALUES")
print(df.isnull().sum().sum())


# 8. DUPLICATE ROWS

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())


# 9. STATISTICAL SUMMARY

print("\nSTATISTICAL SUMMARY")
print(df.describe())


# 10. AVERAGE TRAVEL TIME

print("\nAVERAGE TRAVEL TIME")
print(df["Travel_Time_Minutes"].mean())


# 11. AVERAGE DISTANCE

print("\nAVERAGE DISTANCE")
print(df["Distance_km"].mean())


# 12. AVERAGE SPEED

print("\nAVERAGE SPEED")
print(df["Avg_Speed_kmph"].mean())


# 13. AVERAGE NUMBER OF SIGNALS

print("\nAVERAGE NUMBER OF SIGNALS")
print(df["Number_of_Signals"].mean())


# 14. TRAVEL TIME RANGE

print("\nTRAVEL TIME RANGE")
print("Minimum:", df["Travel_Time_Minutes"].min())
print("Maximum:", df["Travel_Time_Minutes"].max())


# 15. CATEGORICAL DATA ANALYSIS

categorical_columns = [
    "Traffic_Level",
    "Weather",
    "Road_Type",
    "Vehicle_Type",
    "Day_of_Week",
    "Time_of_Day"
]

for column in categorical_columns:
    print("\n", column)
    print(df[column].value_counts(dropna=False))


# 16. TRAVEL TIME BY TRAFFIC LEVEL

print("\nTRAVEL TIME BY TRAFFIC LEVEL")
print(df.groupby("Traffic_Level")["Travel_Time_Minutes"].mean())


# 17. TRAVEL TIME BY WEATHER

print("\nTRAVEL TIME BY WEATHER")
print(df.groupby("Weather")["Travel_Time_Minutes"].mean())


# 18. TRAVEL TIME BY ROAD TYPE

print("\nTRAVEL TIME BY ROAD TYPE")
print(df.groupby("Road_Type")["Travel_Time_Minutes"].mean())


# 19. TRAVEL TIME BY VEHICLE TYPE

print("\nTRAVEL TIME BY VEHICLE TYPE")
print(df.groupby("Vehicle_Type")["Travel_Time_Minutes"].mean())


# 20. TRAVEL TIME BY TIME OF DAY

print("\nTRAVEL TIME BY TIME OF DAY")
print(df.groupby("Time_of_Day")["Travel_Time_Minutes"].mean())


# 21. CORRELATION

numerical_columns = [
    "Distance_km",
    "Avg_Speed_kmph",
    "Number_of_Signals",
    "Travel_Time_Minutes"
]

correlation = df[numerical_columns].corr()

print("\nCORRELATION")
print(correlation)


# ============================================================
# DATA ANALYTICS GRAPHS
# ============================================================


# 22. TRAFFIC LEVEL VS TRAVEL TIME

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Traffic_Level", y="Travel_Time_Minutes",
            order=["Low", "Medium", "High"], palette="crest")
plt.title("Average Travel Time by Traffic Level")
plt.xlabel("Traffic Level")
plt.ylabel("Average Travel Time (Minutes)")
plt.tight_layout()
plt.show()


# 23. WEATHER VS TRAVEL TIME

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Weather", y="Travel_Time_Minutes",
            order=["Clear", "Rainy", "Foggy", "Stormy"], palette="crest")
plt.title("Average Travel Time by Weather")
plt.xlabel("Weather")
plt.ylabel("Average Travel Time (Minutes)")
plt.tight_layout()
plt.show()


# 24. ROAD TYPE VS TRAVEL TIME

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Road_Type", y="Travel_Time_Minutes", palette="crest")
plt.title("Average Travel Time by Road Type")
plt.xlabel("Road Type")
plt.ylabel("Average Travel Time (Minutes)")
plt.tight_layout()
plt.show()


# 25. VEHICLE TYPE VS TRAVEL TIME

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Vehicle_Type", y="Travel_Time_Minutes", palette="crest")
plt.title("Average Travel Time by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Average Travel Time (Minutes)")
plt.tight_layout()
plt.show()


# 26. TIME OF DAY VS TRAVEL TIME

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Time_of_Day", y="Travel_Time_Minutes",
            order=["Morning", "Afternoon", "Evening", "Night"], palette="crest")
plt.title("Average Travel Time by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Average Travel Time (Minutes)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# ============================================================
# EDA STARTS HERE
# ============================================================

print("\n")
print("=" * 50)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 50)


# ============================================================
# 27. HISTOGRAM - DISTANCE
# ============================================================

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Distance_km", kde=True, color="#1C7293")
plt.title("Distribution of Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ============================================================
# 28. HISTOGRAM - AVERAGE SPEED
# ============================================================

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Avg_Speed_kmph", kde=True, color="#1C7293")
plt.title("Distribution of Average Speed")
plt.xlabel("Average Speed (km/h)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ============================================================
# 29. HISTOGRAM - NUMBER OF SIGNALS
# ============================================================

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Number_of_Signals", kde=True, color="#1C7293")
plt.title("Distribution of Number of Signals")
plt.xlabel("Number of Signals")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ============================================================
# 30. HISTOGRAM - TRAVEL TIME
# ============================================================

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Travel_Time_Minutes", kde=True, color="#E09900")
plt.title("Distribution of Travel Time")
plt.xlabel("Travel Time (Minutes)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ============================================================
# 31. BOXPLOTS - ALL NUMERICAL FEATURES (2x2 GRID)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

sns.boxplot(data=df, x="Distance_km", ax=axes[0, 0], color="#5FA8C7")
axes[0, 0].set_title("Distance (km)")

sns.boxplot(data=df, x="Avg_Speed_kmph", ax=axes[0, 1], color="#5FA8C7")
axes[0, 1].set_title("Average Speed (km/h)")

sns.boxplot(data=df, x="Number_of_Signals", ax=axes[1, 0], color="#5FA8C7")
axes[1, 0].set_title("Number of Signals")

sns.boxplot(data=df, x="Travel_Time_Minutes", ax=axes[1, 1], color="#E09900")
axes[1, 1].set_title("Travel Time (Minutes)")

plt.suptitle("Boxplots of Numerical Features", y=1.02, fontsize=13)
plt.tight_layout()
plt.show()


# ============================================================
# 32. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(9, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# 33. DISTANCE VS TRAVEL TIME (colored by Traffic Level)
# ============================================================

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Distance_km", y="Travel_Time_Minutes",
                 hue="Traffic_Level", alpha=0.6, palette="crest")
plt.title("Distance vs Travel Time")
plt.xlabel("Distance (km)")
plt.ylabel("Travel Time (Minutes)")
plt.tight_layout()
plt.show()


# ============================================================
# 34. SPEED VS TRAVEL TIME (colored by Traffic Level)
# ============================================================

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Avg_Speed_kmph", y="Travel_Time_Minutes",
                 hue="Traffic_Level", alpha=0.6, palette="crest")
plt.title("Average Speed vs Travel Time")
plt.xlabel("Average Speed (km/h)")
plt.ylabel("Travel Time (Minutes)")
plt.tight_layout()
plt.show()


# ============================================================
# 35. SIGNALS VS TRAVEL TIME (colored by Road Type)
# ============================================================

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Number_of_Signals", y="Travel_Time_Minutes",
                 hue="Road_Type", alpha=0.6, palette="crest")
plt.title("Number of Signals vs Travel Time")
plt.xlabel("Number of Signals")
plt.ylabel("Travel Time (Minutes)")
plt.tight_layout()
plt.show()


# ============================================================
# 36. TRAFFIC LEVEL VS TRAVEL TIME (BOXPLOT)
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Traffic_Level", y="Travel_Time_Minutes",
            order=["Low", "Medium", "High"], palette="crest")
plt.title("Traffic Level vs Travel Time")
plt.xlabel("Traffic Level")
plt.ylabel("Travel Time (Minutes)")
plt.tight_layout()
plt.show()


# ============================================================
# 37. WEATHER VS TRAVEL TIME (BOXPLOT)
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Weather", y="Travel_Time_Minutes",
            order=["Clear", "Rainy", "Foggy", "Stormy"], palette="crest")
plt.title("Weather vs Travel Time")
plt.xlabel("Weather")
plt.ylabel("Travel Time (Minutes)")
plt.tight_layout()
plt.show()


# ============================================================
# 38. ROAD TYPE VS TRAVEL TIME (BOXPLOT)
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Road_Type", y="Travel_Time_Minutes", palette="crest")
plt.title("Road Type vs Travel Time")
plt.xlabel("Road Type")
plt.ylabel("Travel Time (Minutes)")
plt.tight_layout()
plt.show()


# ============================================================
# 39. VEHICLE TYPE VS TRAVEL TIME (BOXPLOT)
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Vehicle_Type", y="Travel_Time_Minutes", palette="crest")
plt.title("Vehicle Type vs Travel Time")
plt.xlabel("Vehicle Type")
plt.ylabel("Travel Time (Minutes)")
plt.tight_layout()
plt.show()


# ============================================================
# 40. TIME OF DAY VS TRAVEL TIME (BOXPLOT)
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Time_of_Day", y="Travel_Time_Minutes",
            order=["Morning", "Afternoon", "Evening", "Night"], palette="crest")
plt.title("Time of Day vs Travel Time")
plt.xlabel("Time of Day")
plt.ylabel("Travel Time (Minutes)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


print("\nEDA COMPLETE — all charts generated.")
