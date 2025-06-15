import pandas as pd

# Reading from different data sources (uncomment as needed)
# df = pd.read_csv("data/sales_data_sample.csv", encoding="latin1")
# df = pd.read_excel("data/SampleSuperstore.xlsx")
df = pd.read_json("data/sample_Data.json")  # Reading from a JSON file

# Creating a small sample DataFrame from dictionary
data = {
    "Name": ["Ram", "Shyam", "Radha"],
    "Age": [100, 390, 240],
    "City": ["Nagpur", "sirohi", "ayodhya"]
}
newDF = pd.DataFrame(data)

# Saving DataFrame to different file formats
# newDF.to_csv('data/output1.csv', index=False)
# newDF.to_excel('data/output2.xlsx', index=False)
# newDF.to_json('data/output3.json', index=False)

# Reloading sample data
df = pd.read_json("data/sample_Data.json")

print("📌 Displaying the first 10 rows:")
print(df.head(10))

print("📌 Displaying the last 10 rows:")
print(df.tail(10))

print("📌 Dataset Info:")
print(df.info())

# Creating a detailed sample dataset for further operations
sample_data1 = {
    "Name": ["Ram", "Shyam", "Radha", "Sita", "Gita", "Laxman", "Hanuman", "Ravan", "Kumbhkaran", "Vibhishan", "Jatayu"],
    "Age": [100, 390, 240, 200, 150, 300, 400, 500, 600, 700, 800],
    "City": ["Nagpur", "Sirohi", "Ayodhya", "Delhi", "Chennai", "Kolkata", "Bangalore", "Hyderabad", "Pune", "Jaipur", "Lucknow"],
    "Salary": [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000],
    "Performance Score": [90, 85, 80, 95, 70, 75, 60, 55, 50, 45, 40]
}

df = pd.DataFrame(sample_data1)

print("📌 Full Dataset:")
print(df)

print("📌 Descriptive Statistics:")
print(df.describe())

print("📌 Dataset Shape:", df.shape)
print("📌 Column Names:", df.columns.tolist())

# Basic selections
print("📌 Selecting 'Name' and 'Age' columns:")
print(df[["Name", "Age"]])

print("📌 First 5 Rows (using iloc):")
print(df.iloc[0:5])

print("📌 Rows 0 to 5 and 'Name' & 'Age' columns (using loc):")
print(df.loc[0:5, ["Name", "Age"]])

print("📌 Using iloc to select first 5 rows and first two columns:")
print(df.iloc[0:5, [0, 1]])

# Conditional Filtering
print("📌 People with Age > 300:")
print(df[df["Age"] > 300])

print("📌 People with Age > 300 and Salary < 80000:")
print(df[(df["Age"] > 300) & (df["Salary"] < 80000)])

# Sorting
print("📌 Sorting by Age (ascending):")
print(df.sort_values(by="Age"))

print("📌 Sorting by Age (asc) and Salary (desc):")
print(df.sort_values(by=["Age", "Salary"], ascending=[True, False]))

# Adding new column
df["Experience"] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
print("📌 Dataset after adding 'Experience' column:")
print(df)

# Handling Missing Data
sample_data2 = {
    "Name": ["Ram", "Shyam", "Radha", "Sita", "Gita", None, "Hanuman", "Ravan", "Kumbhkaran", "Vibhishan", "Jatayu"],
    "Age": [100, 390, 240, 200, 150, None, 400, 500, 600, 700, 800],
    "City": ["Nagpur", "Sirohi", None, "Delhi", "Chennai", "Kolkata", "Bangalore", None, "Pune", "Jaipur", "Lucknow"],
    "Salary": [10000, 20000, 30000, None, 50000, 60000, 70000, 80000, None, 100000, 110000],
    "Performance Score": [90, None, 80, 95, 70, 75, None, 55, 50, None, 40]
}
df_with_nan = pd.DataFrame(sample_data2)

print("📌 Dataset with missing values:")
print(df_with_nan)

print("📌 Dropping rows with any missing values:")
print(df_with_nan.dropna())

print("📌 Filling missing values with 0:")
print(df_with_nan.fillna(0))

print("📌 Filling missing values with mode:")
print(df_with_nan.fillna(df_with_nan.mode().iloc[0]))

# Renaming Columns
print("📌 Renaming 'Name' → 'Full Name', 'Age' → 'Years':")
print(df.rename(columns={"Name": "Full Name", "Age": "Years"}))

# Dropping columns and rows
print("📌 Dropping 'Experience' column:")
print(df.drop(columns=["Experience"]))

print("📌 Dropping rows at index 0 and 1:")
print(df.drop(index=[0, 1]))

# Changing data types
print("📌 Changing 'Age' column datatype to float:")
df["Age"] = df["Age"].astype(float)
print(df)

# Grouping
print("📌 Average Age grouped by City:")
print(df.groupby("City")["Age"].mean())

# Concatenating DataFrames
data1 = {
    "Name": ["Ram", "Shyam", "Radha"],
    "Age": [100, 390, 240],
    "City": ["Nagpur", "Sirohi", "Ayodhya"]
}
data2 = {
    "Name": ["Sita", "Gita", "Laxman"],
    "Age": [200, 150, 300],
    "City": ["Delhi", "Chennai", "Kolkata"]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

print("📌 Concatenating df1 and df2:")
print(pd.concat([df1, df2], ignore_index=True))

# Merging DataFrames
df3 = pd.DataFrame({
    "Name": ["Ram", "Shyam", "Radha"],
    "Salary": [10000, 20000, 30000]
})
df4 = pd.DataFrame({
    "Name": ["Ram", "Shyam", "Radha"],
    "Performance Score": [90, 85, 80]
})
print("📌 Merging df3 and df4 on 'Name':")
print(pd.merge(df3, df4, on="Name"))

df5 = pd.DataFrame({
    "Name": ["Ram", "Shyam", "Radha"],
    "City": ["Nagpur", "Sirohi", "Ayodhya"]
})
df6 = pd.DataFrame({
    "Name": ["Ram", "Shyam", "Radha"],
    "Age": [100, 390, 240]
})
print("📌 Merging df5 and df6 on 'Name':")
print(df5.merge(df6, on="Name"))
