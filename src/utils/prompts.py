CLASSIFICATION_PROMPT = """
You are an expert AI assistant that is specialized in selecting user preferences.
The task it that you are provided with a prompt and two responses (A and B) to that prompt from different LLMs.
The possible outcomes are 3 classes:
- Winner A: Response A is better
- Winner B: Response B is better
- Tie: Both responses are equally good

The prompt to the models is:
```
{prompt}
```

Then response A is:
```
{response_a}
```

And response B is:
```
{response_b}
```

Based on the above, classify which response is better by choosing one of the following options: "Winner A", "Winner B", or "Tie".
"""
