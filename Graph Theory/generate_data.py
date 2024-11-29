import pandas as pd
import random

# Define the dataset with 50 rows for each column
data = {
    'ID': list(range(1, 51)),
    'Name': [
        "Ali Ahmed", "Sara Khan", "Usman Shah", "Ayesha Malik", "Bilal Tariq", "Fatima Iqbal",
        "Zain Ali", "Mehak Rana", "Imran Hussain", "Nida Shah", "Hamza Khan", "Fariha Azhar",
        "Rizwan Qureshi", "Hina Batool", "Danish Ahmed", "Adeel Sadiq", "Shazia Imran", "Aftab Rehman",
        "Maheen Ali", "Saad Sheikh", "Samina Tariq", "Kamran Khan", "Faizan Malik", "Noreen Iqbal",
        "Anwar Bukhari", "Urooj Javed", "Asim Rehman", "Zoya Shahid", "Farhan Nasir", "Rabia Aslam",
        "Iqra Sohail", "Shahid Khan", "Kashan Ali", "Rabia Iqbal", "Imtiaz Ahmed", "Ayesha Bibi",
        "Bilal Shah", "Hassan Zafar", "Sana Ali", "Iram Sadiq", "Salman Khan", "Amna Rizvi",
        "Zainab Rehman", "Mohsin Javed", "Yasir Ahmed", "Saira Khan", "Tariq Ali", "Faiza Imran",
        "Shabbir Hussain", "Hira Shahid"
    ],
    'Location': [
        'Lahore', 'Karachi', 'Islamabad', 'Lahore', 'Karachi', 'Rawalpindi', 'Faisalabad', 'Multan',
        'Quetta', 'Peshawar', 'Lahore', 'Karachi', 'Islamabad', 'Rawalpindi', 'Faisalabad', 'Lahore',
        'Quetta', 'Multan', 'Peshawar', 'Karachi', 'Rawalpindi', 'Islamabad', 'Lahore', 'Faisalabad',
        'Multan', 'Karachi', 'Rawalpindi', 'Quetta', 'Peshawar', 'Rawalpindi', 'Islamabad', 'Lahore',
        'Karachi', 'Faisalabad', 'Rawalpindi', 'Multan', 'Peshawar', 'Karachi', 'Lahore', 'Islamabad',
        'Faisalabad', 'Rawalpindi', 'Lahore', 'Karachi', 'Faisalabad', 'Peshawar', 'Islamabad', 'Multan',
        'Karachi', 'Rawalpindi'
    ],
    'Friends': []
}
# Generate multiple friends (IDs) for each individual
for person_id in data['ID']:
    # Ensure num_friends is less than or equal to the number of other people
    num_friends = random.randint(1, 5)

    # Get potential friends (excluding the current person)
    potential_friends = [friend_id for friend_id in data['ID'] if friend_id != person_id]

    # Sample friends from potential friends
    friends = random.sample(potential_friends, k=min(num_friends, len(potential_friends)))

    data['Friends'].append(friends)

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
file_path = 'friend_network.csv'
df.to_csv(file_path, index=False)

print(f"CSV file '{file_path}' generated successfully.")
