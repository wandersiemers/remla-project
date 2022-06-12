import random
import threading
import time

import requests

endpoints = ["predict"]
HOST = "http://app:5000/"


def run():
    while True:
        try:
            target = random.choice(endpoints)  # nosec B311
            json = {"title": "Dependency injection in Spring"}
            requests.post(HOST + target, timeout=1, json=json)
        except requests.RequestException:
            print("cannot connect", HOST)
            time.sleep(1)


if __name__ == "__main__":
    for _ in range(4):
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()

    while True:
        time.sleep(1)
