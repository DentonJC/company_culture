import pickle
import wikipedia
import pandas as pd


def get_description(query):
    res = wikipedia.search(query + " Company")
    try:
        if "ADP" in res[0]:
            res[0] = "ADP"  # some wikipedia library bug
        page = wikipedia.page(res[0])
    except wikipedia.DisambiguationError as err:
        page = wikipedia.page(err.options[0])
    except wikipedia.exceptions.PageError:
        return "error"
    time.sleep(1)
    return str(page.content.encode("utf-8"))


df = pd.read_csv("data/culture500_data.csv", encoding="ISO-8859-1")
df = df.set_index(["company", "value"])["sentiment_score"].unstack()

# df.to_csv("data/culture500_data_preprocessed.csv")

companies_names = list(df.index.values)

companies = []
with open("data/companies.csv", "w") as f:
    f.write("name;content\n")
for c in tqdm(companies_names):
    try:
        content = get_description(c)
    except:
        time.sleep(10)
        try:
            content = get_description(c)
        except:
            content = "error"
    companies.append(content)
    with open("data/companies.csv", "a") as f:
        f.write(c + ";" + content.replace(";", " ") + "\n")

with open("data/companies.txt", "wb") as fh:
    pickle.dump(companies, fh)

