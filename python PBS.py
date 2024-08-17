# 3.1. Data cleaning and preprocessing 
#3.1.2. Apply pandas in cleaning and preprocessing data 
import pandas as pd
file_path = 'CustomerTable.csv'  # Đường dẫn đến tệp CSV của bạn
customer_df = pd.read_csv(file_path)  # Đọc tệp CSV vào DataFrame
import matplotlib.pyplot as plt
import pandas as pd

# Giả sử chúng ta đã làm sạch và tải lại file CSV
file_path = 'CustomerTable.csv'
customer_df = pd.read_csv(file_path)


# Đọc dữ liệu từ tệp CSV 1
df = pd.read_csv('CustomerTable.csv')

# Kiểm tra các hàng trùng lặp
duplicate_rows = customer_df[customer_df.duplicated()]

# Hiển thị các hàng trùng lặp nếu có
if not duplicate_rows.empty:
    print("Các hàng trùng lặp được tìm thấy:")
    print(duplicate_rows)
else:
    print("Không có hàng trùng lặp.")


import re

# Các mẫu regex đơn giản để xác thực
email_pattern = r'^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
phone_pattern = r'^\+?\d[\d -]{7,}\d$'  # Định dạng quốc tế và địa phương
postal_pattern = r'^\d{4,10}$'  # Giả sử mã bưu điện là số và có độ dài từ 4 đến 10 chữ số

# Áp dụng xác thực
invalid_emails = customer_df[~customer_df['Email'].apply(lambda x: bool(re.match(email_pattern, x)))]
invalid_phone_numbers = customer_df[~customer_df['PhoneNumber'].apply(lambda x: bool(re.match(phone_pattern, x)))]
invalid_postal_codes = customer_df[~customer_df['PostalCode'].apply(lambda x: bool(re.match(postal_pattern, str(x))))]

# Hiển thị kết quả
print("Email không hợp lệ:\n", invalid_emails)
print("\nSố điện thoại không hợp lệ:\n", invalid_phone_numbers)
print("\nMã bưu điện không hợp lệ:\n", invalid_postal_codes)


# Remove invalid phone numbers
customer_df_cleaned = customer_df[customer_df['PhoneNumber'].apply(lambda x: bool(re.match(phone_pattern, x)))]

# Display the cleaned DataFrame
customer_df_cleaned.head()


# Remove the record with the invalid postal code
customer_df_cleaned = customer_df_cleaned[customer_df_cleaned['PostalCode'].apply(lambda x: bool(re.match(postal_pattern, str(x))))]

# Display the cleaned DataFrame
customer_df_cleaned.head()


# Save the cleaned data to a new CSV file
cleaned_file_path = '/mnt/data/Cleaned_CustomerTable.csv'
customer_df_cleaned.to_csv(cleaned_file_path, index=False)

cleaned_file_path






#3.2. Visualisation  
import matplotlib.pyplot as plt
import pandas as pd

# Load the cleaned CSV file
file_path = '/mnt/data/Cleaned_CustomerTable.csv'
customer_df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime if it exists in the dataset
if 'Date' in customer_df.columns:
    customer_df['Date'] = pd.to_datetime(customer_df['Date'])

# Bar Chart: Distribution of Customers by Region
if 'Region' in customer_df.columns:
    customer_counts_by_region = customer_df['Region'].value_counts()
    
    plt.figure(figsize=(10, 6))
    customer_counts_by_region.plot(kind='bar', color='skyblue')
    plt.title('Number of Customers by Region')
    plt.xlabel('Region')
    plt.ylabel('Number of Customers')
    plt.show()

# Pie Chart: Market Share of Product Groups
if 'ProductGroup' in customer_df.columns:
    product_group_share = customer_df['ProductGroup'].value_counts()
    
    plt.figure(figsize=(8, 8))
    product_group_share.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Market Share of Product Groups')
    plt.ylabel('')
    plt.show()

# Line Chart: Sales Trends Over Time
if 'Date' in customer_df.columns and 'Sales' in customer_df.columns:
    sales_over_time = customer_df.groupby('Date')['Sales'].sum()
    
    plt.figure(figsize=(10, 6))
    sales_over_time.plot(kind='line', color='green')
    plt.title('Sales Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.show()

# Histogram: Distribution of Customer Ages
if 'Age' in customer_df.columns:
    plt.figure(figsize=(10, 6))
    plt.hist(customer_df['Age'], bins=15, color='orange', edgecolor='black')
    plt.title('Distribution of Customer Ages')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()




#Hồi quy tuyến tính (Linear Regression)
import pandas as pd
import numpy as np

# Đọc dữ liệu từ file CSV đã tải lên
file_path = '/mnt/data/CustomerTable.csv'
data = pd.read_csv(file_path)

# Giả định tạo cột 'Sales' (doanh số) bằng một số ngẫu nhiên cho ví dụ minh họa
np.random.seed(42)  # Đảm bảo kết quả có thể tái lập
data['Sales'] = np.random.randint(1000, 5000, size=len(data))

# Chọn các biến đầu vào (ví dụ 'CustomerID' và 'PostalCode')
X = data[['CustomerID', 'PostalCode']]
y = data['Sales']


#Sử dụng thư viện Scikit-Learn (Sklearn)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

#Dự đoán doanh số trong tương lai (Predict Future Sales)
from sklearn.metrics import mean_squared_error, r2_score

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

