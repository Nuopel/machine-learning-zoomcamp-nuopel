## a new customer informations
customer = {
  'customerid': '8879-zkjof',
  'gender': 'female',
  'seniorcitizen': 0,
  'partner': 'no',
  'dependents': 'no',
  'tenure': 41,
  'phoneservice': 'yes',
  'multiplelines': 'no',
  'internetservice': 'dsl',
  'onlinesecurity': 'yes',
  'onlinebackup': 'no',
  'deviceprotection': 'yes',
  'techsupport': 'yes',
  'streamingtv': 'yes',
  'streamingmovies': 'yes',
  'contract': 'one_year',
  'paperlessbilling': 'yes',
  'paymentmethod': 'bank_transfer_(automatic)',
  'monthlycharges': 79.85,
  'totalcharges': 3320.75
}

#full deploy
# import requests ## to use the POST method we use a library named requests
# url = 'http://192.168.1.97:9696/predict'
# response = requests.post(url, json=customer) ## post the customer information in json format
# result = response.json() ## get the server response
# print(result)

# deploy dev local
import requests ## to use the POST method we use a library named requests
url = 'http://127.0.0.1:9696/predict'
response = requests.post(url, json=customer) ## post the customer information in json format
result = response.json() ## get the server response
print(result)


