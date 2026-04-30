from talkie_evals.summary import markdown_table


def test_arithmetic_markdown_table():
    result = {
        "dataset": "EleutherAI/arithmetic",
        "models": [
            {
                "model_name": "talkie-1930-13b-it",
                "tasks": [
                    {
                        "name": "arithmetic_2da",
                        "description": "2-digit addition",
                        "summary": {
                            "accuracy": 0.5,
                            "correct_count": 1,
                            "total": 2,
                            "first_token_greedy_accuracy": 0.5,
                        },
                    }
                ],
            }
        ],
    }
    table = markdown_table(result)
    assert "arithmetic_2da" in table
    assert "50.0%" in table
