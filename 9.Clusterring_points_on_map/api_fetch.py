import requests
import pandas as pd
import io
urlData = requests.get('https://api.data.gov.in/resource/bccc6a91-cde0-4d1a-b255-6aab90a9e303?api-key=579b464db66ec23bdd00000104d25d16a7284b96728a24b871ae1380&format=json&limit=1000')

data = urlData.json()
df = pd.DataFrame(data['records'])
print(df)

specific_column=df[['latitude___n','longitude___e']]

print("Specific Columns:")
print(specific_column)
specific_column.insert(0, 'line_number', range(1, len(specific_column) + 1))
specific_column.to_csv('specific_column.txt', sep=' ', index = False, header= True)