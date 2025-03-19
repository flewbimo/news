

import sys
import os
from dotenv import load_dotenv
from crew import *
import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


def run():
    inputs = {
        'news': "特朗普在2024年美国总统大选中获胜"
    }
    NewsCrew.kickoff(inputs)


if __name__ == "__main__":
    run()
