
import sys
import os
from dotenv import load_dotenv

from crew import NewsCrew

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

def run():
    inputs = {
        'news': "John McCain opposed bankruptcy protections for families \"who were only in bankruptcy because of medical expenses they couldn't pay.\"",
        'newsset': ["\"Bennie Thompson actively cheer-led riots in the \u201990s.\"",
                    "Says\u00a0Maggie Hassan was \"out of state on 30 days over the last three months.\"",
                    "\"BUSTED: CDC Inflated COVID Numbers, Accused of Violating Federal Law\"",
                    "I'm the only (Republican)\u00a0candidate that has actually reduced the size of government.\"",
                    "\"There are actually only 30 countries that practice birthright citizenship.\""
        ]
    }
    NewsCrew.kickoff(inputs)


if __name__ == "__main__":
    run()