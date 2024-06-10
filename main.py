import numpy as np
import pandas as pd
import gspread
import torch
from sentence_transformers import SentenceTransformer
from google.oauth2.service_account import Credentials

#import the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def compute_similarity(x,y):
      embedding_1= model.encode(x, convert_to_tensor=True)
      embedding_2 = model.encode(y, convert_to_tensor=True)
      cosi = torch.nn.CosineSimilarity(dim=0)
      output = (cosi(embedding_1, embedding_2)+1)/2
      return output

#Accessing the data in the sheet
doc_id="16zMI3339L9uZFGRG275w7DZ0-TlPMd4f8KDdDlrQQuA"
tab_name="Sheet1"
url = "https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}".format(doc_id, tab_name)[:-1]
projects=pd.read_csv(url)

#Google Sheet API
scopes = ["https://www.googleapis.com/auth/spreadsheets"] 

creds = Credentials.from_service_account_info({
    "type": "service_account",
    "project_id": "gpdata-425213",
    "private_key_id": "3d24201e9a778e21104080701c22c83df250d599",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDGOVa2WzeZ850q\nCZNKuyvJ4yQLB+ztYc1bUhH44VNy4+2yqCm7c/6zr8Sm9rXZ1c04JIQsKVJuV1q5\nAInzlHtd508AdMKjHE3/DYrXN1082bw/KsrvS7Gw+y7Wobw1LXMN+jGxOAag1A9O\nJ9qw9QGs951qcVIo7yAotYsBYr7l+CioTLTOyKXGIIBa3FCUV4mwYDXRU56iZ3rJ\n0HTG18nHwVkojpewFFgJXm/s/17I7twdlbWc66nUgyyYaNGbhz963GHzT7HRQ6N0\nzq3/oW7pKj3sRlMhNqm2PrW4dZErI0VNC+lEs5HL+JogEmHaLl9IfJaHFWh9Jmtp\n7glIncFbAgMBAAECggEAVkey9KDzrLEbIacVU5vwzWlu0NZcTF8bsbohVIFPf1yy\no+cKcytkG/ZP1JTiRrUHJH7QgSBjXt/q/0e6xClHIXKDMqGf7rttP603V8IjdU8Y\n3y3TMyFxcKWEl6vCbisgP189rfPC/tIO72ftisS6O/1zqVc1+ddL2ixbGvlOOm5r\nJiZk7fc70nCInKD4qWfqQBPK6EyeKmcO6Ntar44wcfo5kIwn3EOk1NMdB/nG5yNf\nrsZwtpi4TbYwKzobK6twDeSwJxu2JoJ9h+ooObJTMdN4OHcJO2K+HM+oLlBEANHY\nISA3947rhIrC34Z0bTH8+oGg52KXxFy8fU3Yv8M5UQKBgQDzLObGUs3QuqqKMvsP\niH2fdl2vLhXft8Gq7WudUgOIDQY6tdw55Hm3IECBhnfQ1Lt/y23vfAnst/2/hNsF\n1GDld8vCLkAGyUpjTYl03a3f/TwddTmObRMj2U8WZ2+wuCA62eY1e1aCNnrs7dfY\ntGt2+p06k1W/RxZU0EOF+EvNLwKBgQDQrYzsj4GgrIKEHkf63Tc7g/5Ml10P0ccq\noJ93NUiQMu0gr0aLvWSqWOs6WVSAQD8Aehb6i/xdw9Cq3EmyGHgcE9us7YlP3lhg\novl6pWbMQBpVae+ol6wwBOjUMXKNZEJxyezipvdWO/YLcQAvetDVfVvfnQ/PDlst\nZQCFmMK7lQKBgCQCrTYHQxU631BR1l3pf3jixWLQt0qG4rYWLI6Ce6VlEFwXXEJy\ndBfLPeIwcIPLTOzSyjfhrXKRmJEI8oo9dg/lGpZp1O9sVYi5DbbxsPLvhDx0hI5z\n1pbDcnPF44NO8O7mH7IhzqC/wppdak5cAWIAINJwyQznUQZERQuMxmTfAoGAEgJo\nXyIjddJtkSlr3OKqmaBSmhmWFn9sSOmD2a3njUpX3LJDzFuUDH+QDYEYIdlplojy\n4ryiExWLNLO+SHiEJSgxlUMKzrHJvs1R6pvLu2Ts4OI7pLkySxKhZW6/DCTS9y8O\nGqF+Rxr1qRcfhPl8fHBNNYdAjgYXKFvHJ8B/TeECgYEAoRvnytoEDFVvtcC6QqDg\ngaaka9O2rUd/6tGVJL6qgCvXR0iIC5lrOmdfdDJffPWPQf1vsUdwyRKNMhf8AxPp\nWUD9CoLxVQNp+2G70Wll96uq0cbHUVcBAovPcWhkkdlJh32JIMfQ7L0wvwnKRKQe\n3lBiYc9w7zzRpLUhSWlIlnw=\n-----END PRIVATE KEY-----\n",
    "client_email": "gp-api@gpdata-425213.iam.gserviceaccount.com",
    "client_id": "110108541011026371376",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/gp-api%40gpdata-425213.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
},scopes=scopes)

client = gspread.authorize(creds)

def append_data_to_sheet(new_idea):
   sheet = client.open_by_key(doc_id).sheet1
   #get the last cell in the sheet
   last_row_values = sheet.get_all_values()[-1]
   last_number = int(last_row_values[0])
   ID = int(last_number) + 1
   new_data = [ID,new_idea]
   sheet.append_row(new_data)
   return f"Data appended successfully!"

def remove_data(search_value):
  sheet = client.open_by_key(doc_id).sheet1
  all_values = sheet.get_all_values()
  row_index = None
  for i, row in enumerate(all_values):
        if row[1] == search_value:
            row_index = i 
  if row_index is not None:
        del all_values[row_index]
        sheet.clear()  
        sheet.update(all_values)
        msg="Project data was removed successfully!"
  else:
        msg="No such project found."
  return msg


from fastapi import FastAPI, File, UploadFile, HTTPException
app=FastAPI()

@app.get("/")
def root():
    return{"Similarity checker"}

@app.post("/similarity")
def search_match(idea):
  max_score = 0
  most_similar_project_index = None
  for i in range(len(projects["Idea"])):
    score = compute_similarity(idea, str(projects.loc[i, "Idea"])) * 100
    if score > max_score:
      max_score = score
      most_similar_project_index = i

  if max_score > 75:
    msg = "Match Found"
    data = {
      "match": projects.loc[most_similar_project_index, "Idea"],
      "score": round(float(max_score), 2)
    }
  elif 70 <= max_score <= 75:  
    msg = "Neutral"
    data = {
      "match": projects.loc[most_similar_project_index, "Idea"],
      "score": round(float(max_score), 2)
    }
  else:
    msg = "No match"
    data = None  

  return msg, data

@app.get("/add")
def append_data_route(new_idea):
    try:
        result_message = append_data_to_sheet(new_idea)
        return {"message": result_message}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/delete")
def remove_project(idea):
   try:
      result_message = remove_data(idea)
      return {"message": result_message}
   except Exception as e:
      return {"error": str(e)}
