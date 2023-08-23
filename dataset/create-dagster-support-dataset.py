import json
from pathlib import Path
from typing import Dict, List

import langchain
import numpy as np
import typer
from langchain.cache import SQLiteCache
from langchain.llms import OpenAI
from tqdm import tqdm

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


def _is_daster_empl(title: str) -> bool:
    return "Elementl" in title or "Dagster" in title


def get_daster_empl(users_path: str):
    with open(users_path) as f:
        users = json.load(f)
    dagster_users_by_title_id = [u["id"] for u in users if _is_daster_empl(u["profile"]["title"])]
    return dagster_users_by_title_id


def add_gpt4_replies(result: List[Dict[str, str]]) -> List[Dict[str, str]]:
    p = """

    You are dagster expert.

    Question form the user: 
    '''
    {q}
    '''
    List of replies: 
    '''
    {r}
    '''

    Here is one sentence answer based on info above: 
    """

    llm = OpenAI(temperature=0.1, model_name="gpt-4-32k")
    for x in tqdm(result):
        prompt = p.format(q=x["question"], r=x["replies"])
        gpt4_one_liner = llm(prompt)
        x["gpt4_replies_target"] = gpt4_one_liner
    return result


def add_dagster_empl(result: List[Dict[str, str]]) -> List[Dict[str, str]]:
    for m in result:
        replies = m["replies"]
        is_dagster_empl = np.array(m["is_dagster_empl"])
        if is_dagster_empl.any():
            fist_dagster_user = None
            last_dagster_user = None

        first_index = np.argmax(is_dagster_empl)
        last_index = len(is_dagster_empl) - np.argmax(is_dagster_empl[::-1]) - 1

        fist_dagster_user = replies[first_index]
        last_dagster_user = replies[last_index]

        m["dagster_empl_first_target"] = fist_dagster_user
        m["dagster_empl_last_target"] = last_dagster_user
    return result


def create_datasets(
    directory_path: str = "dagster-slack/dagster-support/",
    users_path: str = "dagster-slack/users.json",
    output_path: str = "dagster-support-dataset.json",
):
    # Directory path
    directory_path = Path(directory_path)
    users_path = Path(users_path)

    # List all JSON files in the directory
    json_files = list(directory_path.glob("*.json"))
    print(f"Total json_files = {len(json_files)}")
    json_files = [x for x in json_files if "2023" in x.name]
    print(f"Total json_files from 2023 = {len(json_files)}")

    # get all ts to text
    ts2text = {}
    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)
            for m in data:
                ts2text[m["ts"]] = m["text"]

    # get all dagster users
    daster_empl = set(get_daster_empl(users_path=users_path))

    result = []
    # Process each JSON file
    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)

        data_with_reactions = [m for m in data if "reactions" in m and "replies" in m]
        data_with_reactions_solved = [
            m for m in data_with_reactions if "dagster-bot-resolve" in [x["name"] for x in m["reactions"]]
        ]

        for m in data_with_reactions_solved:
            question = m["text"]
            replies = [ts2text[x["ts"]] for x in m["replies"]]
            is_dagster_empl = [x["user"] in set(daster_empl) for x in m["replies"]]

            result.append({"question": question, "replies": replies, "is_dagster_empl": is_dagster_empl})

    print(f"Total samples {len(result)}")
    result = add_gpt4_replies(result=result)
    result = add_dagster_empl(result=result)

    with open(output_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    typer.run(create_datasets)
