import requests

url = "https://data.tradefeeds.com/api/v1/company_ratings"
params = {
    "key": "nuh-uh youre not getting my key",
    "ticker": "MSFT",
    "date_from": "2017-01-01",
    "date_to": "2024-01-01",
}
headers = {
    "accept": "application/json"
}

response = requests.get(url, params=params, headers=headers)

# Print the JSON data
if response.status_code == 200:
    data = response.json()
    import pandas as pd

    df = pd.DataFrame(data)
    print(df)
    
    # Save the JSON data to a file
    with open('company_ratings.json', 'w') as f:
        pd.DataFrame(data).to_json(f, orient='records', indent=4)

    # Save the raw response to a file
    with open('company_ratings_raw_response.json', 'w') as f:
        f.write(response.text)
else:
    print(f"Failed to retrieve data: {response.status_code}")